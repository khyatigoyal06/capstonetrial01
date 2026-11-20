"""Target-location split and robust summaries. No frame-level train/test split."""
import numpy as np

# Twenty-five distributed points on coordinates different from the initial pilot.
# The five final test points cover all four quadrants plus the center. Validation
# spans the center/top and each quadrant; training includes both screen edges.
TARGETS = [(x,y) for y in (.08,.29,.50,.71,.92) for x in (.08,.29,.50,.71,.92)]
TRAIN_IDX=tuple(i for i in range(len(TARGETS)) if i not in {0,2,4,6,8,12,16,18,20,24})
VALID_IDX=(2,6,8,16,18)
TEST_IDX=(0,4,12,20,24)

def robust_summary(samples):
    a=np.asarray(samples,float)
    if a.ndim!=2 or len(a)==0: raise ValueError("No valid feature samples")
    return np.nanmedian(a,axis=0)

def target_pixel(ratio,size):
    """Map normalized [0,1] target to the drawable window's pixel coordinates."""
    w,h=size
    if w<=0 or h<=0: raise ValueError("Invalid drawable size")
    return (round(ratio[0]*(w-1)),round(ratio[1]*(h-1)))
