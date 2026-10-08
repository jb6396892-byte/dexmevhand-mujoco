"""Legacy GLX streaming adapter: hidden double-buffer context, MuJoCo FBO output."""
def stream_context(sim):
    import mujoco_py
    import glfw
    original = mujoco_py.cymj.GlfwContext

    for context in sim.render_contexts:
        if context.offscreen and isinstance(context.opengl_context,original):
            context.opengl_context.make_context_current()
            return context

    class OwnedRenderContext(mujoco_py.MjRenderContext):
        def __del__(self):
            # The legacy Cython destructor frees GL buffers in whichever context is current.
            if self.opengl_context is not None:
                self.opengl_context.make_context_current()

    class HiddenDoubleBufferedContext(original):
        def _create_window(self, offscreen, quiet=False):
            glfw.default_window_hints()
            glfw.window_hint(glfw.VISIBLE,0)
            glfw.window_hint(glfw.DOUBLEBUFFER,1)
            glfw.window_hint(glfw.SAMPLES,0)
            window = glfw.create_window(self._INIT_WIDTH,self._INIT_HEIGHT,'MuJoCo frame worker',None,None)
            if not window:
                raise RuntimeError('Hidden double-buffered GLX context creation failed')
            return window

    # Only the worker-local window factory changes; the frozen binary and physics do not.
    mujoco_py.cymj.GlfwContext = HiddenDoubleBufferedContext
    warmstart=sim.data.qacc_warmstart.copy()
    try:
        return OwnedRenderContext(sim,offscreen=True,opengl_backend='glfw',quiet=True)
    finally:
        sim.data.qacc_warmstart[:]=warmstart
        mujoco_py.cymj.GlfwContext = original
