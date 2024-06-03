#pragma once

#include <glm/glm.hpp>

struct CameraFrame {
    CameraFrame(float worldScale);

    glm::vec3 getPOI() const;

    /*! re-compute all orientation related fields from given
      'user-style' camera parameters */
    void setOrientation(/* camera origin    : */ glm::vec3 origin,
                        /* point of interest: */ glm::vec3 poi,
                        /* up-vector        : */ glm::vec3 up);

    /*! tilt the frame around the z axis such that the y axis is "facing upwards" */
    void forceUpFrame();

    void setUpVector(glm::vec3 up);

    float computeStableEpsilon(float f) const;

    float computeStableEpsilon(const glm::vec3 v) const;

    glm::vec3 get_from() const;

    glm::vec3 get_at() const;
    glm::vec3 get_up() const;

    // linear3f frame{one};
    glm::mat3 frame = glm::mat3(1.0f);
    glm::vec3 position{0, -1, 0};
    /*! distance to the 'point of interst' (poi); e.g., the point we
      will rotate around */
    float poiDistance{1.f};
    glm::vec3 upVector{0, 1, 0};
    /* if set to true, any change to the frame will always use to
       upVector to 'force' the frame back upwards; if set to false,
       the upVector will be ignored */
    bool forceUp{true};

    /*! multiplier how fast the camera should move in world space
      for each unit of "user specifeid motion" (ie, pixel
      count). Initial value typically should depend on the world
      size, but can also be adjusted. This is actually something
      that should be more part of the manipulator widget(s), but
      since that same value is shared by multiple such widgets
      it's easiest to attach it to the camera here ...*/
    float motionSpeed{1.f};

    /*! gets set to true every time a manipulator changes the camera
      values */
    bool modified{true};
};

