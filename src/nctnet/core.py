from __future__ import annotations

import argparse, math
from dataclasses import dataclass, replace
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
try:
    import yaml
except Exception:
    yaml = None


@dataclass
class NCTConfig:
    vocab_size:int=4096; max_seq_len:int=128; d_model:int=128; n_layers:int=6; n_heads:int=4; d_ff:int=512
    memory_slots:int=16; memory_dim:int=128; relation_slots:int=16; relation_dim:int=128
    n_regimes:int=8; n_cards:int=6; coherence_rank:int=8; residual_bottleneck:int=32
    beta_coh:float=0.5; coh_clip_min:float=-5.0; coh_clip_max:float=5.0
    up_gain:float=0.1; mem_gain:float=0.1; rel_gain:float=0.1; adm_gain:float=0.1; coh_gain:float=0.0
    loss_up:float=0.05; loss_mem:float=0.05; loss_rel:float=0.02; loss_adm:float=0.02; loss_res:float=0.02
    loss_cards:float=0.01; loss_regime:float=0.01; loss_reversibility:float=0.02
    def validate(self):
        if self.d_model % 4: raise ValueError('d_model must be divisible by 4')
        if self.memory_dim != self.d_model: raise ValueError('v0.1 requires memory_dim == d_model')
        if self.relation_dim != self.d_model: raise ValueError('v0.1 requires relation_dim == d_model')

@dataclass
class NCTState:
    memory:torch.Tensor; relations:torch.Tensor; slow_ctx:torch.Tensor
    def detach(self): return NCTState(self.memory.detach(), self.relations.detach(), self.slow_ctx.detach())
    def to(self, device): return NCTState(self.memory.to(device), self.relations.to(device), self.slow_ctx.to(device))
    def replace(self, **kw): return replace(self, **kw)


def split_up(x): return x.chunk(2, dim=-1)
def merge_up(u,p): return torch.cat([u,p], dim=-1)
def rms(x, eps=1e-6): return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True)+eps)
def spos(x, eps=1e-6): return F.softplus(x)+eps
def center(x): return x - x.mean(dim=tuple(range(x.dim()-1)), keepdim=True)

class MLP(nn.Module):
    def __init__(self, i,h,o, zero_last=False):
        super().__init__(); self.net=nn.Sequential(nn.Linear(i,h), nn.GELU(), nn.Linear(h,o))
        if zero_last: nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)
    def forward(self,x): return self.net(x)

class AdditiveCouplingBlock(nn.Module):
    def __init__(self, d_half, hidden, gain=0.1):
        super().__init__(); self.f=MLP(d_half,hidden,d_half); self.g=MLP(d_half,hidden,d_half); self.gain=gain
    def forward(self,x):
        u,p=split_up(x); u1=u+self.gain*self.f(rms(p)); p1=p+self.gain*self.g(rms(u1)); return merge_up(u1,p1)
    def inverse(self,y):
        u1,p1=split_up(y); p=p1-self.gain*self.g(rms(u1)); u=u1-self.gain*self.f(rms(p)); return merge_up(u,p)

class UPReciprocity(nn.Module):
    def __init__(self,d,hidden):
        super().__init__(); self.p2u=MLP(d//2,hidden,d//2); self.u2p=MLP(d//2,hidden,d//2); self.metric=nn.Parameter(torch.zeros(d))
    def forward(self,z):
        u,p=split_up(z); uh=self.p2u(p); ph=self.u2p(u); err=torch.cat([u-uh,p-ph],-1); energy=(err.pow(2)*spos(self.metric)).mean(); return err, energy

def sinusoidal_slot_basis(slots, dim, device=None):
    i=torch.arange(slots,device=device,dtype=torch.float32).unsqueeze(1); k=torch.arange((dim+1)//2,device=device,dtype=torch.float32).unsqueeze(0)+1
    th=2*math.pi*i/max(slots,1); b=torch.stack([torch.sin(k*th),torch.cos(k*th)],-1).reshape(slots,-1)[:,:dim]; return rms(b)
def ring_laplacian(slots, device=None):
    A=torch.zeros(slots,slots,device=device)
    for i in range(slots): A[i,(i-1)%slots]=1; A[i,(i+1)%slots]=1
    return torch.diag(A.sum(1))-A

class TopologicalMemory(nn.Module):
    def __init__(self, slots, dim):
        super().__init__(); self.slots=slots; self.dim=dim; self.q=nn.Linear(dim,dim); self.w=nn.Linear(dim,dim); self.drive=nn.Linear(dim,dim); self.gate=nn.Linear(2*dim,1)
        self.register_buffer('keys', sinusoidal_slot_basis(slots,dim), persistent=False); self.register_buffer('lap', ring_laplacian(slots), persistent=False)
    def initial_state(self,B,device=None): return 0.01*self.keys.to(device).unsqueeze(0).expand(B,-1,-1).contiguous()
    def forward(self,z,M):
        B,S,D=z.shape; zm=z.mean(1); keys=self.keys.to(z.device); keyed=M+0.05*keys.unsqueeze(0)
        a=torch.softmax(torch.einsum('bd,bfd->bf', self.q(zm), keyed)/math.sqrt(D), -1); read=torch.einsum('bf,bfd->bd',a,M); summary=M.mean(1)
        w=self.w(zm); b=torch.softmax(torch.einsum('bd,bfd->bf',w,keyed)/math.sqrt(D),-1); write=b.unsqueeze(-1)*(w.unsqueeze(1)+0.05*keys.unsqueeze(0)); diff=-torch.einsum('ij,bjd->bid', self.lap.to(z.device), M)
        need=torch.sigmoid(self.gate(torch.cat([zm,summary],-1))).view(B,1,1); Mn=rms(M+need*0.05*write+0.01*diff); drv=self.drive(read).unsqueeze(1).expand(B,S,D); en=(zm-read).pow(2).mean()+0.01*Mn.pow(2).mean()
        return read, summary, Mn, drv, en

class RelationBank(nn.Module):
    def __init__(self, slots, dim):
        super().__init__(); self.slots=slots; self.dim=dim; self.trace=nn.Linear(2*dim,dim); self.drive=nn.Linear(dim,dim)
        self.register_buffer('keys', sinusoidal_slot_basis(slots,dim), persistent=False); self.register_buffer('lap', ring_laplacian(slots), persistent=False)
    def initial_state(self,B,device=None): return 0.01*self.keys.to(device).unsqueeze(0).expand(B,-1,-1).contiguous()
    def forward(self,z,R,mem_read):
        B,S,D=z.shape; rw=self.trace(torch.cat([z.mean(1),mem_read],-1)); keyed=R+0.05*self.keys.to(z.device).unsqueeze(0)
        a=torch.softmax(torch.einsum('bd,bed->be',rw,keyed)/math.sqrt(D),-1); read=torch.einsum('be,bed->bd',a,R); write=a.unsqueeze(-1)*(rw.unsqueeze(1)+0.05*self.keys.to(z.device).unsqueeze(0)); diff=-torch.einsum('ij,bjd->bid', self.lap.to(z.device), R)
        Rn=rms(R+0.04*write+0.01*diff); drv=self.drive(read).unsqueeze(1).expand(B,S,D); en=(rw-read).pow(2).mean()+0.01*Rn.pow(2).mean(); return read,Rn.mean(1),Rn,drv,en

class RegimeController(nn.Module):
    def __init__(self,d,R):
        super().__init__(); self.ctx=MLP(4*d,d,d); self.logits=nn.Linear(4*d,R)
    def forward(self,zm,ms,rs,prev=None,task=None):
        if prev is None: prev=torch.zeros_like(zm)
        if task is None: task=torch.zeros_like(zm)
        c=rms(0.8*prev+0.2*self.ctx(torch.cat([zm,ms,rs,prev+task],-1))); p=torch.softmax(self.logits(torch.cat([zm,ms,rs,c],-1)),-1); ent=-(p*(p+1e-8).log()).sum(-1).mean(); sw=(c-prev).pow(2).mean(); return c,p,ent,sw

class AdmissibilityGate(nn.Module):
    def __init__(self,d,R):
        super().__init__(); self.gate=MLP(4*d+R,d,d); self.drive=nn.Linear(d,d); nn.init.constant_(self.gate.net[-1].bias,2.0)
    def forward(self,z,ctx,mem,rel,probs):
        B,S,D=z.shape; cat=torch.cat([z,ctx[:,None].expand(B,S,D),mem[:,None].expand(B,S,D),rel[:,None].expand(B,S,D),probs[:,None].expand(B,S,probs.shape[-1])],-1)
        g=torch.sigmoid(self.gate(cat)); safe=rms(z+0.1*mem[:,None]+0.1*rel[:,None]); za=g*z+(1-g)*safe; drv=self.drive(za-z); en=(g.mean()-0.85).pow(2)+(g.var(unbiased=False)-0.02).pow(2)+0.01*(za-z).pow(2).mean(); return za,g,drv,en

class ResidualProjector(nn.Module):
    def __init__(self,d,b): super().__init__(); self.obs=nn.Linear(d,b); self.rec=nn.Linear(b,d)
    def forward(self,z):
        o=self.obs(z); zh=self.rec(o); nu=z-zh; return nu, nu.pow(2).mean()

class CausalCoherenceTensor(nn.Module):
    def __init__(self,d,rank,R,beta=0.5,cmin=-5,cmax=5):
        super().__init__(); self.diag=nn.Parameter(torch.zeros(d)); self.low=nn.Parameter(torch.randn(d,rank)*0.02); self.rdiag=nn.Parameter(torch.zeros(R,d)); self.beta=beta; self.cmin=cmin; self.cmax=cmax; self.w=nn.Parameter(torch.ones(6))
    def forward(self,z,up,mem,rel,adm,res,probs):
        xc=center(z); dims=tuple(range(xc.dim()-1)); var=xc.pow(2).mean(dims); diag=spos(self.diag)+0.1*torch.matmul(probs,F.softplus(self.rdiag)).mean(0); idiag=(var*diag).sum(); proj=xc@self.low; ilow=proj.var(dim=tuple(range(proj.dim()-1)),unbiased=False).sum(); ww=F.softplus(self.w)
        terms=torch.stack([1/(1+up),1/(1+mem),1/(1+rel),1/(1+adm),1/(1+res),probs.var(-1,unbiased=False).mean()+0.1]); info=idiag+ilow+(ww*terms).sum(); mass=torch.exp(self.beta*torch.clamp(info/max(z.shape[-1],1),self.cmin,self.cmax)).reshape(1); en=(up+mem+rel+adm+0.25*res)/mass.detach().clamp_min(1e-6); return mass,info,en,idiag,ilow

class MultiCardReadout(nn.Module):
    def __init__(self,d,K,out):
        super().__init__(); self.K=K; self.cards=nn.ModuleList([MLP(2*d,d,d) for _ in range(K)]); self.sel=MLP(3*d,d,K); self.heads=nn.ModuleList([nn.Linear(d,out) for _ in range(K)])
    def forward(self,z,ctx,mem):
        B,S,D=z.shape; c=ctx[:,None].expand(B,S,D); cards=torch.stack([rms(m(torch.cat([z,c],-1))) for m in self.cards],2); w=torch.softmax(self.sel(torch.cat([z.mean(1),mem,ctx],-1)),-1); outs=torch.stack([head(cards[:,:,k,:]) for k,head in enumerate(self.heads)],2); mixed=torch.einsum('bk,bskv->bsv',w,outs); ent=-(w*(w+1e-8).log()).sum(-1).mean(); target=0.65*torch.log(torch.tensor(float(self.K),device=z.device)); load=((w.mean(0)-1/self.K)**2).sum(); en=(ent-target).pow(2)+load; return mixed,w,en,ent

class CoherenceDrive(nn.Module):
    def __init__(self,d,R): super().__init__(); self.net=MLP(4*d+R+1,d,d,zero_last=True)
    def forward(self,z,mass,mem,rel,ctx,probs):
        B,S,D=z.shape; return self.net(torch.cat([z,mem[:,None].expand(B,S,D),rel[:,None].expand(B,S,D),ctx[:,None].expand(B,S,D),probs[:,None].expand(B,S,probs.shape[-1]),mass.reshape(1,1,1).expand(B,S,1)],-1))

class NCTBlock(nn.Module):
    def __init__(self,cfg:NCTConfig):
        super().__init__(); cfg.validate(); d=cfg.d_model; self.cfg=cfg; self.rev=AdditiveCouplingBlock(d//2,cfg.d_ff,cfg.up_gain); self.memory=TopologicalMemory(cfg.memory_slots,d); self.relations=RelationBank(cfg.relation_slots,d); self.regime=RegimeController(d,cfg.n_regimes); self.adm=AdmissibilityGate(d,cfg.n_regimes); self.up=UPReciprocity(d,cfg.d_ff); self.res=ResidualProjector(d,cfg.residual_bottleneck); self.coh=CausalCoherenceTensor(d,cfg.coherence_rank,cfg.n_regimes,cfg.beta_coh,cfg.coh_clip_min,cfg.coh_clip_max); self.cdrive=CoherenceDrive(d,cfg.n_regimes); self.norm=nn.LayerNorm(d)
    def initial_state(self,B,device=None): return NCTState(self.memory.initial_state(B,device), self.relations.initial_state(B,device), torch.zeros(B,self.cfg.d_model,device=device))
    def set_coh_gain(self,v): self.cfg.coh_gain=float(v)
    def forward(self,z,state:NCTState,task=None):
        zb=self.rev(z); rev=(z-self.rev.inverse(zb)).abs().mean(); mr,ms,Mn,md,me=self.memory(zb,state.memory); rr,rs,Rn,rd,re=self.relations(zb,state.relations,mr); ctx,prob,rent,sw=self.regime(zb.mean(1),ms,rs,state.slow_ctx,task); za,gate,ad,ae=self.adm(zb,ctx,mr,rr,prob); err,ue=self.up(za); nu,ne=self.res(za); mass,info,ce,idiag,ilow=self.coh(za,ue,me,re,ae,ne,prob); cd=self.cdrive(za,mass,mr,rr,ctx,prob); zn=self.norm(rms(za+self.cfg.mem_gain*md+self.cfg.rel_gain*rd+self.cfg.adm_gain*ad+self.cfg.coh_gain*mass.view(1,1,1)*cd)); aux={'up_energy':ue,'mem_energy':me,'rel_energy':re,'adm_energy':ae,'residue_energy':ne,'regime_energy':sw,'coh_energy':ce,'mass':mass.mean(),'gate_mean':gate.mean(),'regime_entropy':rent,'reversibility_error':rev,'cards_energy':torch.zeros((),device=z.device),'card_entropy':torch.zeros((),device=z.device)}; return zn,state.replace(memory=Mn,relations=Rn,slow_ctx=ctx),aux

class NCTLanguageModel(nn.Module):
    def __init__(self,cfg:NCTConfig):
        super().__init__(); cfg.validate(); self.cfg=cfg; self.tok=nn.Embedding(cfg.vocab_size,cfg.d_model); self.pos=nn.Embedding(cfg.max_seq_len,cfg.d_model); self.blocks=nn.ModuleList([NCTBlock(cfg) for _ in range(cfg.n_layers)]); self.readout=MultiCardReadout(cfg.d_model,cfg.n_cards,cfg.vocab_size); self.norm=nn.LayerNorm(cfg.d_model)
    def initial_state(self,B,device=None):
        s=[b.initial_state(B,device) for b in self.blocks]; return NCTState(torch.stack([x.memory for x in s]).mean(0),torch.stack([x.relations for x in s]).mean(0),torch.stack([x.slow_ctx for x in s]).mean(0))
    def set_coh_gain(self,v): self.cfg.coh_gain=float(v); [b.set_coh_gain(v) for b in self.blocks]
    def disable_coherence(self): self.set_coh_gain(0.0)
    def forward(self,input_ids,labels=None,state=None):
        B,S=input_ids.shape
        if S>self.cfg.max_seq_len: raise ValueError('sequence exceeds max_seq_len')
        if state is None: state=self.initial_state(B,input_ids.device)
        h=self.tok(input_ids)+self.pos(torch.arange(S,device=input_ids.device)[None].expand(B,S)); aux=[]
        for b in self.blocks: h,state,a=b(h,state); aux.append(a)
        h=self.norm(h); logits,w,ce,cent=self.readout(h,state.slow_ctx,state.memory.mean(1)); aux[-1]={**aux[-1],'cards_energy':ce,'card_entropy':cent}
        loss=None
        if labels is not None:
            task=F.cross_entropy(logits.reshape(-1,logits.size(-1)),labels.reshape(-1)); struct=sum(self.cfg.loss_up*a['up_energy']+self.cfg.loss_mem*a['mem_energy']+self.cfg.loss_rel*a['rel_energy']+self.cfg.loss_adm*a['adm_energy']+self.cfg.loss_res*a['residue_energy']+self.cfg.loss_cards*a['cards_energy']+self.cfg.loss_regime*a['regime_energy']+self.cfg.loss_reversibility*a['reversibility_error'] for a in aux)/len(aux); loss=task+struct
        metrics={k:float(torch.stack([a[k].detach().float().cpu() for a in aux]).mean()) for k in aux[0].keys()}
        return type('NCTLMOutput',(),{'loss':loss,'logits':logits,'state':state,'aux':aux,'metrics':metrics})()

class SyntheticLMDataset(Dataset):
    def __init__(self,vocab_size=256,seq_len=32,size=1024): self.vocab_size=vocab_size; self.seq_len=seq_len; self.size=size
    def __len__(self): return self.size
    def __getitem__(self,idx):
        task=idx%4
        if task==0: seq=[(10+idx+i)%self.vocab_size for i in range(self.seq_len+1)]
        elif task==1: seq=([20,21,22,21]*((self.seq_len+4)//4))[:self.seq_len+1]
        elif task==2: seq=[30+(i%2) for i in range(self.seq_len+1)]
        else: seq=[40 if i%5==0 else 8+(idx+i)%20 for i in range(self.seq_len+1)]
        return {'input_ids':torch.tensor(seq[:-1]),'labels':torch.tensor(seq[1:])}

def collate(items): return {'input_ids':torch.stack([x['input_ids'] for x in items]), 'labels':torch.stack([x['labels'] for x in items])}

@dataclass
class TrainConfig:
    run_dir:str='runs/debug'; steps:int=1; batch_size:int=2; lr:float=5e-4; seq_len:int=16; seed:int=0; coh_warmup:int=1; coh_ramp:int=2; coh_target:float=0.02

def train_tiny_lm(model_cfg:NCTConfig, train_cfg:TrainConfig):
    torch.manual_seed(train_cfg.seed); Path(train_cfg.run_dir).mkdir(parents=True,exist_ok=True); model=NCTLanguageModel(model_cfg); ds=SyntheticLMDataset(min(model_cfg.vocab_size,256),train_cfg.seq_len,max(32,train_cfg.steps*train_cfg.batch_size)); loader=DataLoader(ds,batch_size=train_cfg.batch_size,shuffle=True,collate_fn=collate); opt=torch.optim.AdamW(model.parameters(),lr=train_cfg.lr); it=iter(loader)
    for step in tqdm(range(train_cfg.steps),desc='train',leave=False):
        try: batch=next(it)
        except StopIteration: it=iter(loader); batch=next(it)
        gain=0.0 if step<train_cfg.coh_warmup else min(train_cfg.coh_target,train_cfg.coh_target*(step-train_cfg.coh_warmup+1)/max(1,train_cfg.coh_ramp)); model.set_coh_gain(gain); out=model(batch['input_ids'],labels=batch['labels']); out.loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); opt.zero_grad(set_to_none=True)
    path=Path(train_cfg.run_dir)/'best.pt'; torch.save({'model':model.state_dict(),'config':model_cfg.__dict__},path); return path

def _load_cfg(path):
    if yaml is None: return {}
    with open(path,'r',encoding='utf-8') as f: return yaml.safe_load(f) or {}

def main():
    p=argparse.ArgumentParser('nctnet'); sub=p.add_subparsers(required=True)
    t=sub.add_parser('train-lm'); t.add_argument('--config',required=True)
    i=sub.add_parser('inspect'); i.add_argument('--checkpoint',required=True); i.add_argument('--prompt',default='1 2 3 4')
    a=sub.add_parser('ablate'); a.add_argument('--checkpoint',required=True); a.add_argument('--ablation',default='no_coherence')
    args=p.parse_args()
    if hasattr(args,'config'):
        raw=_load_cfg(args.config); ck=train_tiny_lm(NCTConfig(**raw.get('model',{})),TrainConfig(**raw.get('train',{}))); print(f'saved={ck}')
    elif hasattr(args,'prompt'):
        ck=torch.load(args.checkpoint,map_location='cpu'); cfg=NCTConfig(**ck['config']); model=NCTLanguageModel(cfg); model.load_state_dict(ck['model']); toks=[int(x)%cfg.vocab_size for x in args.prompt.split() if x.lstrip('-').isdigit()] or [1,2,3,4]; out=model(torch.tensor(toks).unsqueeze(0)); print('logits_shape',tuple(out.logits.shape)); print('top_next',torch.topk(out.logits[0,-1],min(5,cfg.vocab_size)).indices.tolist()); [print(f'{k}: {v:.6f}') for k,v in out.metrics.items()]
    else:
        ck=torch.load(args.checkpoint,map_location='cpu'); cfg=NCTConfig(**ck['config']); model=NCTLanguageModel(cfg); model.load_state_dict(ck['model']); ids=torch.tensor([[1,2,3,4,5,6,7,8]]); full=model(ids).logits; model.disable_coherence(); abl=model(ids).logits; print('ablation',args.ablation); print('mean_abs_delta',float((full-abl).abs().mean()))
