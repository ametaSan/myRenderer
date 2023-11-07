#ifndef __MODEL_H__
#define __MODEL_H__
#include <vector>
#include <string>
#include "geometry.h"
#include "tgaimage.h"

class Model {
private:
    std::vector<Vec3f> verts_;
    std::vector<std::vector<Vec3i>> faces_; // attention, this Vec3i means vertex/uv/normal
    
    // faces_ seperately corresponding to different objects
    std::vector<std::vector<std::vector<Vec3i>>> obj_faces_;
    std::vector<std::string> obj_names_;

    std::vector<Vec3f> norms_;
    std::vector<Vec2f> uv_;
    // TGAImage diffusemap_;
    std::vector<TGAImage> obj_diffusemaps_; // textures for different objects
    TGAImage normalmap_;
    TGAImage specularmap_;
    int obj_size = 0;
    void load_texture(std::string filename, const char *suffix, TGAImage &img);
public:
    Model(const char *filename);
    ~Model();
    int nverts();
    int nfaces();
    int objectnfaces(int iobject);
    int nobjects();
    Vec3f normal(int iface, int nthvert);
    Vec3f objnormal(int iobject, int iface, int nthvert);
    Vec3f normal(Vec2f uv);
    Vec3f vert(int i);
    Vec3f vert(int iface, int nthvert);
    Vec3f objvert(int iobject, int iface, int nthvert);
    Vec2f uv(int iface, int nthvert);
    Vec2f objectuv(int iobject, int iface, int nthvert);
    TGAColor diffuse(int iobject, Vec2f uv);
    float specular(Vec2f uv);
    std::vector<int> face(int idx);
    std::vector<int> objectface(int iobject, int iface);
};
#endif //__MODEL_H__

