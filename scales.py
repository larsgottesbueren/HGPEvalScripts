import matplotlib.ticker as ticker
import matplotlib.scale as mscale
import matplotlib.transforms as mtransforms
import numpy as np

class CubeRootScale(mscale.ScaleBase):
    name = 'cuberoot'

    def __init__(self, axis, **kwargs):
        mscale.ScaleBase.__init__(self, axis, **kwargs)

    def set_default_locators_and_formatters(self, axis):
        axis.set_major_locator(ticker.AutoLocator())
        axis.set_major_formatter(ticker.ScalarFormatter())
        axis.set_minor_locator(ticker.NullLocator())
        axis.set_minor_formatter(ticker.NullFormatter())

    def limit_range_for_scale(self, vmin, vmax, minpos):
        return  max(0., vmin), vmax

    class CubeRootTransform(mtransforms.Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True

        def transform_non_affine(self, a): 
            x = np.array(a)
            return np.sign(x) * (np.abs(x)**(1.0/3.0))

        def inverted(self):
            return CubeRootScale.InvertedCubeRootTransform()

    class InvertedCubeRootTransform(mtransforms.Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True

        def transform(self, a):
            x = np.array(a)
            return np.sign(x) * (np.abs(x)**3)

        def inverted(self):
            return CubeRootScale.CubeRootTransform()

    def get_transform(self):
        return self.CubeRootTransform()

mscale.register_scale(CubeRootScale)

class FifthRootScale(mscale.ScaleBase):
    name = 'fifthroot'

    def __init__(self, axis, **kwargs):
        mscale.ScaleBase.__init__(self, axis, **kwargs)

    def set_default_locators_and_formatters(self, axis):
        axis.set_major_locator(ticker.AutoLocator())
        axis.set_major_formatter(ticker.ScalarFormatter())
        axis.set_minor_locator(ticker.NullLocator())
        axis.set_minor_formatter(ticker.NullFormatter())

    def limit_range_for_scale(self, vmin, vmax, minpos):
        return  max(0., vmin), vmax

    class FifthRootTransform(mtransforms.Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True

        def transform_non_affine(self, a): 
            x = np.array(a)
            return np.sign(x) * (np.abs(x)**0.2)

        def inverted(self):
            return FifthRootScale.InvertedFifthRootTransform()

    class InvertedFifthRootTransform(mtransforms.Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True

        def transform(self, a):
            x = np.array(a)
            return np.sign(x) * (np.abs(x)**5)

        def inverted(self):
            return FifthRootScale.FifthRootTransform()

    def get_transform(self):
        return self.FifthRootTransform()

mscale.register_scale(FifthRootScale)
