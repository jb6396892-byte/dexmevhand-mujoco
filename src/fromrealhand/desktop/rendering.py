"""Legacy GLX streaming adapter: hidden double-buffer context, MuJoCo FBO output."""
def stream_context(sim):
    import mujoco_py
    import glfw
    original = mujoco_py.cymj.GlfwContext

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
    try:
        return mujoco_py.MjRenderContext(sim,offscreen=True,opengl_backend='glfw',quiet=True)
    finally:
        mujoco_py.cymj.GlfwContext = original
