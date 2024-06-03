#pragma once

#include "CameraFrameManip.h"

// ------------------------------------------------------------------
/*! camera manipulator with the following traits

    - left button rotates the camera around the viewer position

    - middle button strafes in camera plane

    - right buttton moves forward/backwards

  */
struct FlyModeManip : public CameraFrameManip {
    FlyModeManip(CameraFrame* cameraFrame);

private:
    /*! helper function: rotate camera frame by given degrees, then
      make sure the frame, poidistance etc are all properly set,
      the widget gets notified, etc */
    virtual void rotate(float deg_u, float deg_v) override;

    /*! helper function: move forward/backwards by given multiple of
      motion speed, then make sure the frame, poidistance etc are
      all properly set, the widget gets notified, etc */
    virtual void move(float step) override;
};
