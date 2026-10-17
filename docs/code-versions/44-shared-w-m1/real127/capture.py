"""Observe the original graph only. No candidate alternative is built here."""
class Capture:
    def __init__(self, model, layers, T):
        self.layers,self.T=layers,T
        self.names={id(op):name for name,op in model.named_modules()}
        self.records=[];self.originals=[];self.active=False
    def install(self):
        for cls,method,kind in ((self.layers.Linear,'__call__','Linear'),(self.layers.Embedding,'as_linear','Embedding.as_linear')):
            original=getattr(cls,method)
            def call(op,x,_original=original,_kind=kind):
                y=_original(op,x)
                if self.active and x.ndim==2 and x.shape[0]==self.T:
                    self.records.append({'order':len(self.records),'name':self.names[id(op)],'kind':_kind,'op':op,'x':x,'y':y})
                return y
            self.originals.append((cls,method,original));setattr(cls,method,call)
    def close(self):
        self.active=False
        for cls,method,original in reversed(self.originals):setattr(cls,method,original)
        self.originals.clear()
