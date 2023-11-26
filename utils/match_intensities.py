import SimpleITK as sitk

def match_intensities(ref, mov) :
    matcher = sitk.HistogramMatchingImageFilter()
    matcher.SetNumberOfHistogramLevels(256)
    matcher.SetNumberOfMatchPoints(15)
    matcher.SetThresholdAtMeanIntensity(True)
    return matcher.Execute(mov, ref) 