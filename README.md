# Person Detectors

A collection of person detectors used for person detection, tracking and
related image processing research and applications. At the moment this
repo contains person detectors:

- Harr cascade based person detector
- HOG based person detector
- Blob person detector
- Mask RCNN (Object detection + segmentation based) person detector
- Tensorflow Object detection based person detector
- OpenPose tensorflow implementation based person detector

The main objective of this repository is to allow comparisons to be
done on different person detectors.

## Example

```
python DetectorPreview.py <path to test video> <detector>
```

`detector` can be one of **haar, blob, mask, op, tfod, tfop ** and
**hog**.

## Contributors

Many thanks to [Madhawa Vidanapathirana](https://github.com/madhawav) for
working on evaluating these detectors.


