#include <iostream>
#include "tgaimage.h"
#include "mygl.h"



int main(int argc, char** argv)
{
	//init_model("model/arona_BlueArchive.obj");
	init_model("model1/mitsuki.obj");
	
	// env->setLightDir((Vec3f(-2, 0, -1).normalize()));
	env->setLightDir((Vec3f(-1, 0, 0).normalize()));

	shadowBufferParamsPtr->setPerspTag(false);
	shadowBufferParamsPtr->lookAt_SB(Vec3f(0, 10, 0), env->getLightDir());
	shadowBufferParamsPtr->init_Matrices();
	shadowBufferParamsPtr->init_zbuffer();
	TGAImage depth(shadowBufferParamsPtr->width, shadowBufferParamsPtr->height, TGAImage::RGB);
	DepthShader depthshader;
	solidBackground(depth, black);
	drawModel(depthshader, model, depth, shadowBufferParamsPtr);
	depth.flip_vertically();
	depth.write_tga_file("depth.tga");

	frameBufferParamsPtr->setPerspTag(true);
	frameBufferParamsPtr->lookAt_FB(Vec3f(0, 10, 0), Vec3f(2, 10, 10));
	frameBufferParamsPtr->init_Matrices();
	frameBufferParamsPtr->init_zbuffer();
	TGAImage frame(frameBufferParamsPtr->width, frameBufferParamsPtr->height, TGAImage::RGB);
	//GouraudShader shader;
	PhongShader shader;
	solidBackground(frame, light_blue);
	drawModel(shader, model, frame, frameBufferParamsPtr);
	drawInit(frame);
	frame.flip_vertically();
	frame.write_tga_file("output.tga");

	delete model;
	shadowBufferParamsPtr->del_zbuffer();
	frameBufferParamsPtr->del_zbuffer();
	delete shadowBufferParamsPtr;
	delete frameBufferParamsPtr;
	delete env;
	return 0;
}
