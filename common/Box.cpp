#include "Box.h"

void Box::extend(const glm::vec3& point)
{
    min = glm::min(min, point);
    max = glm::max(max, point);
}
