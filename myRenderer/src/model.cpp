#include <iostream>
#include <fstream>
#include <sstream>
#include "model.h"

Model::Model(const char *filename) : verts_(), faces_(), norms_(), uv_(), 
obj_faces_(), obj_names_(), obj_diffusemaps_(), normalmap_(), specularmap_() {
    std::ifstream in;
    in.open (filename, std::ifstream::in);
    if (in.fail()) return;
    std::string line;
    std::vector<std::vector<Vec3i>> tmp_faces_; // present processed faces
    std::string obj_name;   // present processed name
    
    while (!in.eof()) {
        std::getline(in, line);
        std::istringstream iss(line.c_str());
        char trash;
        if (!line.compare(0, 2, "v ")) {
            iss >> trash;
            Vec3f v;
            for (int i=0;i<3;i++) iss >> v[i];
            verts_.push_back(v);
        } else if (!line.compare(0, 3, "vn ")) {
            iss >> trash >> trash;
            Vec3f n;
            for (int i=0;i<3;i++) iss >> n[i];
            norms_.push_back(n);
        } else if (!line.compare(0, 3, "vt ")) {
            iss >> trash >> trash;
            Vec2f uv;
            for (int i=0;i<2;i++) iss >> uv[i];
            uv_.push_back(uv);
        } else if (!line.compare(0, 2, "g ")) {
            if (tmp_faces_.size() == 0) {
                obj_name = line.substr(2);
                obj_names_.push_back(obj_name);
            }      
            if (tmp_faces_.size() != 0) {
                obj_faces_.push_back(tmp_faces_);
                std::cout << "# faces of obj " << obj_name << ": " << tmp_faces_.size() << std::endl;
                std::cout << "index: " << obj_faces_.size() << std::endl;
                tmp_faces_.clear();

                obj_name = line.substr(2);
                obj_names_.push_back(obj_name);
            }
        } else if (!line.compare(0, 2, "f ")) {
            std::vector<Vec3i> f;
            Vec3i tmp;
            iss >> trash;
            while (iss >> tmp[0] >> trash >> tmp[1] >> trash >> tmp[2]) {
                for (int i=0; i<3; i++) tmp[i]--; // in wavefront obj all indices start at 1, not zero
                f.push_back(tmp);
            }
            faces_.push_back(f);
            tmp_faces_.push_back(f);
        }  
    }

    // process the last object
    obj_faces_.push_back(tmp_faces_);
    tmp_faces_.clear();
    std::cerr << "# v# " << verts_.size() << " f# "  << faces_.size() << " vt# " << uv_.size() << " vn# " << norms_.size() << std::endl;

    obj_size = obj_names_.size();
    obj_diffusemaps_.resize(obj_size);
    for (int i = 0; i < obj_size; i++)
    {
        std::string obj_name = obj_names_[i];
        std::cout<< obj_name << std::endl;
        load_texture("model_arona/" + obj_name, ".tga", obj_diffusemaps_[i]);
        std::cout << "Index of textures: " << i << std::endl;
    }

    //load_texture("model_yoimiya/yoimiya_cloth", ".tga", diffusemap_);
    //load_texture(filename, ".tga",      normalmap_);
    //load_texture(filename, ".tga",    specularmap_);
}

Model::~Model() {}

int Model::nverts() {
    return (int)verts_.size();
}

int Model::nfaces() {
    return (int)faces_.size();
}

int Model::objectnfaces(int iobject) {
    return (int)obj_faces_[iobject].size();
}

int Model::nobjects() {
    return (int)obj_size;
}

std::vector<int> Model::face(int idx) {
    std::vector<int> face;
    for (int i=0; i<(int)faces_[idx].size(); i++) face.push_back(faces_[idx][i][0]);    // here size == 3 always
    return face;
}

std::vector<int> Model::objectface(int iobject, int iface) {
    std::vector<int> face;
    for (int i = 0; i < (int)obj_faces_[iobject][iface].size(); i++)  // here size == 3 always
        face.push_back(obj_faces_[iobject][iface][i][0]);   
    return face;
}

Vec3f Model::vert(int i) {
    return verts_[i];
}

Vec3f Model::vert(int iface, int nthvert) {
    return verts_[faces_[iface][nthvert][0]];
}

Vec3f Model::objvert(int iobject, int iface, int nthvert) {
    return verts_[obj_faces_[iobject][iface][nthvert][0]];
}

void Model::load_texture(std::string filename, const char *suffix, TGAImage &img) {
    std::string texfile(filename);
    size_t dot = texfile.find_last_of(".");
    if (1) {
        texfile = texfile.substr(0,dot) + std::string(suffix);
        std::cerr << "texture file " << texfile << " loading " << (img.read_tga_file(texfile.c_str()) ? "ok" : "failed") << std::endl;
        img.flip_vertically();
    }
}

TGAColor Model::diffuse(int iobject, Vec2f uvf) {
    Vec2i uv(uvf[0]* obj_diffusemaps_[iobject].get_width(), uvf[1] * obj_diffusemaps_[iobject].get_height());
    return obj_diffusemaps_[iobject].get(uv[0], uv[1]);
}

Vec3f Model::normal(Vec2f uvf) {
    Vec2i uv(uvf[0]*normalmap_.get_width(), uvf[1]*normalmap_.get_height());
    TGAColor c = normalmap_.get(uv[0], uv[1]);
    Vec3f res;
    for (int i=0; i<3; i++)
        res[2-i] = (float)c[i]/255.f*2.f - 1.f;
    return res;
}

Vec2f Model::uv(int iface, int nthvert) {
    return uv_[faces_[iface][nthvert][1]];
}

Vec2f Model::objectuv(int iobject, int iface, int nthvert) {
    return uv_[obj_faces_[iobject][iface][nthvert][1]];
}

float Model::specular(Vec2f uvf) {
    Vec2i uv(uvf[0]*specularmap_.get_width(), uvf[1]*specularmap_.get_height());
    return specularmap_.get(uv[0], uv[1])[0]/1.f;
}

Vec3f Model::normal(int iface, int nthvert) {
    int idx = faces_[iface][nthvert][2];
    return norms_[idx].normalize();
}

Vec3f Model::objnormal(int iobject, int iface, int nthvert) {
    int idx = obj_faces_[iobject][iface][nthvert][2];
    return norms_[idx].normalize();
}

