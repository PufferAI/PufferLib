#include "rocket_lander.h"

int main(void) {
    demo();
    return 0;
}



    /*
    Entity legs[2] = {0};
    for (int i = 0; i < 2; i++) {
        float leg_i = (i == 0) ? -1 : 1;
        b2Vec2 leg_extent = (b2Vec2){LEG_W / SCALE, LEG_H / SCALE};

        b2BodyDef leg = b2DefaultBodyDef();
        leg.type = b2_dynamicBody;
        leg.position = (b2Vec2){-leg_i * LEG_AWAY, INITIAL_Y - LANDER_HEIGHT/2 - leg_extent.y/2};
        //leg.position = (b2Vec2){0, 0};
        leg.rotation = b2MakeRot(leg_i * 1.05);
        b2BodyId leg_id = b2CreateBody(world_id, &leg);

        b2Polygon leg_box = b2MakeBox(leg_extent.x, leg_extent.y);
        b2ShapeDef leg_shape = b2DefaultShapeDef();
        b2CreatePolygonShape(leg_id, &leg_shape, &leg_box);

        float joint_x = leg_i*LANDER_WIDTH/2;
        float joint_y = INITIAL_Y - LANDER_HEIGHT/2 - leg_extent.y/2;
        b2Vec2 joint_p = (b2Vec2){joint_x, joint_y};

        b2RevoluteJointDef joint = b2DefaultRevoluteJointDef();
        joint.bodyIdA = lander_id;
        joint.bodyIdB = leg_id;
        joint.localAnchorA = b2Body_GetLocalPoint(lander_id, joint_p);
        joint.localAnchorB = b2Body_GetLocalPoint(leg_id, joint_p);
        joint.localAnchorB = (b2Vec2){i * 0.5, LEG_DOWN};
        joint.enableMotor = true;
        joint.enableLimit = true;
        joint.maxMotorTorque = LEG_SPRING_TORQUE;
        joint.motorSpeed = 0.3*i;

        if (i == 0) {
            joint.lowerAngle = 40 * DEGTORAD;
            joint.upperAngle = 45 * DEGTORAD;
        } else {
            joint.lowerAngle = -45 * DEGTORAD;
            joint.upperAngle = -40 * DEGTORAD;
        }

        b2JointId joint_id = b2CreateRevoluteJoint(world_id, &joint);

        legs[i] = (Entity){
            .extent = leg_extent,
            .bodyId = leg_id,
        };
    }
    */


