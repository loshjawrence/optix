#include "Box.h"

void Box::extend(const glm::vec3& point)
{
    min = glm::min(min, point);
    max = glm::max(max, point);
}

glm::vec3 Box::diag() const
{
    return max - min;
}

glm::vec3 Box::center() const
{
    return min + (diag() * 0.5f);
}
