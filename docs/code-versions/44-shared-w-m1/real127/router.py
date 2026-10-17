"""Injected standalone local-test routing. No runtime import, default OFF."""
class ProjectionRouter:
    def __init__(self, mx, shared, fallback, runtime):
        self.mx,self.shared,self.fallback,self.runtime=mx,shared,fallback,runtime
        self.enabled=False
    def set_enabled(self, enabled):
        if type(enabled) is not bool:raise TypeError('strict bool required')
        self.enabled=enabled
    def project(self, op, x, *, weight_strides, scale_strides, x_strides):
        if not self.enabled:return self.fallback(op,x,self.mx),{'route':'qualified113_fallback','reason':'default_OFF'}
        n,k=op.weight.shape[0],x.shape[-1]
        # Only declared geometry/layout eligibility failures choose fallback.
        # Compilation, arithmetic and prepared-operand failures propagate STOP.
        try:
            self.shared.validate_geometry(n,k,weight_strides,scale_strides)
            self.shared.validate_input(x.shape,x_strides,k)
        except ValueError as error:
            return self.fallback(op,x,self.mx),{'route':'qualified113_fallback','reason':str(error)}
        prepared=self.shared.prepare(self.mx,op,weight_strides=weight_strides,scale_strides=scale_strides,**self.runtime)
        if prepared.weight is not op.weight or prepared.scales is not op.scales:raise RuntimeError('prepared operands changed')
        y=prepared.project(x,observed_x_strides=x_strides)
        return y,{'route':'shared124','reason':'declared_geometry_T_and_observed_layout','same_original_weight_and_scales':True}
