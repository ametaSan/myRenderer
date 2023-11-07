#include <iostream>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <limits>
#include "model.h"
#include "geometry.h"
#include "tgaimage.h"
#include <Windows.h>

using std::shared_ptr;
using std::make_shared;

// TGAColors
TGAColor black = TGAColor(0, 0, 0, 255);
TGAColor white = TGAColor(255, 255, 255, 255);
TGAColor red = TGAColor(255, 0, 0, 255);
TGAColor green = TGAColor(0, 255, 0, 255);
TGAColor blue = TGAColor(0, 0, 255, 255);
TGAColor light_blue = TGAColor(189, 246, 255, 255);
TGAColor gray = TGAColor(128, 128, 128, 255);
TGAColor light_green = TGAColor(133, 214, 183, 255);

// constants
const float PI = 3.14159265358979323846;
const float infinity = std::numeric_limits<float>::infinity();
const float lowest = std::numeric_limits<float>::lowest();

// Utility Functions
void setFramePixelColor(HDC& hdc, int x, int y, TGAColor& color, int height)
{
	SetPixel(hdc, x, height-y-1, RGB(color[2], color[1], color[0]));
}


// line-plane intersection
struct LineSegment
{
	Vec3f p1;
	Vec3f p2;

	LineSegment(Vec3f _p1, Vec3f _p2) : p1(_p1), p2(_p2) {}
};
struct Plane
{
	Vec3f base;
	Vec3f normal;

	Plane(Vec3f _base, Vec3f _normal) :base(_base), normal(_normal) {}
};
inline bool pointPlaneNormalSide(const Vec3f& p, const Plane& pl)
{
	return (((p - pl.base) * pl.normal) > -0.00001f);
}
bool lineIntersectPlane(const LineSegment& ls, const Plane& pl, Vec3f& ret)
{
	if (pointPlaneNormalSide(ls.p1, pl) == pointPlaneNormalSide(ls.p2, pl))return false;

	float t = (((ls.p1 - pl.base) * pl.normal) / ((ls.p2 - ls.p1) * pl.normal)) * (-1.f);
	ret = ls.p1 + (ls.p2 - ls.p1) * t;
	return true;
}


struct Environment
{
private:
	// Vec3f light_pos;
	Vec3f light_dir;
	float ambientIntensity;

public:
	void setLightDir(Vec3f _light_dir) { light_dir = _light_dir; }
	Vec3f getLightDir()const { return light_dir; }
	void setAmbientIntensity(float _ambientIntensity) { ambientIntensity = _ambientIntensity; }
	float getAmbientIntensity()const { return ambientIntensity; }

	Environment()
	{
		setLightDir((Vec3f(0, 0, -1).normalize()));
		setAmbientIntensity(10.f);
	}
};
//Environment env;
// frame info for object based rendering
struct BufferParams
{
public:

	// screen
	const int width = 1280;
	const int height = 720;

	// camera box
	const int l = -width / 80;
	const int r = width / 80;
	const int b = -height / 80;
	const int t = height / 80;
	const int n = -8;
	const int f = -64;

	// zbuffer
	float** zbuffer = NULL;
	void initZbuffer()
	{
		// note here, rotated by 90 degrees
		zbuffer = new float* [width];
		for (int x = 0; x < width; x++)
			zbuffer[x] = new float[height];

		for (int x = 0; x < width; x++)
			for (int y = 0; y < height; y++)
				zbuffer[x][y] = lowest;
	}
	void delZbuffer()
	{
		for (int x = 0; x < width; x++)
			delete[] zbuffer[x];

		delete[] zbuffer;
	}

	// perspective params: true -> persp; false -> ortho
	bool persp_tag = false;
	void setPerspTag(bool _tag) { persp_tag = _tag; }

	// global params
	Vec3f top = Vec3f(0, -1, 0);	// top pointer of camera
	Vec3f center;
	Vec3f eye;
	Vec3f gaze_dir;
	void lookAt_SB(Vec3f _center, Vec3f _light_dir)
	{
		center = _center;
		eye = _center + (_light_dir * (-10.f));	// light position
		gaze_dir = (_center - eye).normalize();
	}
	void lookAt_FB(Vec3f _center, Vec3f _eye)
	{
		center = _center;
		eye = _eye;
		gaze_dir = (_center - eye).normalize();
	}

	// Orthogonal: (Modeling tf) -> Camera tf -> (Ortho) Proj tf -> Viewport tf
	// Perspective: (Modeling tf) -> Camera tf -> (Pers) Proj tf -> Viewport tf
	mat<4, 4, float> m_vp;
	mat<4, 4, float> m_proj;
	mat<4, 4, float> m_cam;
	mat<4, 4, float> M_viewport()
	{
		mat<4, 4, float> m = Matrix::identity();

		m[0][0] = width / 2.f;
		m[1][1] = height / 2.f;
		m[0][3] = (width - 1.f) / 2.f;
		m[1][3] = (height - 1.f) / 2.f;

		return m;
	};
	mat<4, 4, float> M_project()
	{
		mat<4, 4, float> m = Matrix::identity();
		if (persp_tag)	// perspective
		{
			// bugs may lives here.., the sign of outputs doesn't matter
			m[0][0] = (float)(2 * n) / (float)(r - l);
			m[0][2] = (float)(l + r) / (float)(l - r);
			m[1][1] = (float)(2 * n) / (float)(t - b);
			m[1][2] = (float)(b + t) / (float)(b - t);
			m[2][2] = (float)(f + n) / (float)(n - f);
			m[2][3] = (float)(2 * f * n) / (float)(f - n);
			m[3][2] = 1.f;
			m[3][3] = 0.f;

			m = m * Matrix::negative_identity();

			// perspective matrix supported by chat-gpt
			//m[0][0] = (float)(2 * n) / (float)(r - l);
			//m[0][2] = (float)(l + r) / (float)(r - l);
			//m[1][1] = (float)(2 * n) / (float)(t - b);
			//m[1][2] = (float)(b + t) / (float)(t - b);
			//m[2][2] = (float)(f + n) / (float)(n - f);
			//m[2][3] = (float)(2 * f * n) / (float)(n - f);
			//m[3][2] = -1.f;
			//m[3][3] = 0.f;
		}
		else	// orthogonal
		{
			m[0][0] = 2.f / (r - l);
			m[1][1] = 2.f / (t - b);
			m[2][2] = 2.f / (n - f);
			m[0][3] = (float)(r + l) / (float)(l - r);
			m[1][3] = (float)(t + b) / (float)(b - t);
			m[2][3] = (float)(n + f) / (float)(f - n);
		}
		return m;
	};
	mat<4, 4, float> M_camera()
	{
		// t always points to up
		// center is the gazed point
		Vec3f w = gaze_dir * (-1.f);
		Vec3f u = cross(top, w).normalize();
		Vec3f v = cross(w, u);

		Matrix m1 = Matrix::identity();
		Matrix m2 = Matrix::identity();

		m1[0][0] = u.x;
		m1[0][1] = u.y;
		m1[0][2] = u.z;
		m1[1][0] = v.x;
		m1[1][1] = v.y;
		m1[1][2] = v.z;
		m1[2][0] = w.x;
		m1[2][1] = w.y;
		m1[2][2] = w.z;

		m2[0][3] = -eye.x;
		m2[1][3] = -eye.y;
		m2[2][3] = -eye.z;

		return m1 * m2;
	};

	// other functions
	Vec3f world2screen(Vec4f world_coord) const
	{
		Vec4f _screen_coord = (m_vp * m_proj * m_cam) * world_coord;
		Vec3f screen_coord = proj<3>(_screen_coord / _screen_coord[3]);
		return screen_coord;
	}

	Vec3f world2CVV(Vec4f world_coord) const
	{
		Vec4f _NDC = (m_proj * m_cam) * world_coord;
		Vec3f NDC = proj<3>(_NDC / _NDC[3]);
		return NDC;
	}

	Vec3f CVV2screen(Vec4f NDC) const
	{
		Vec4f _screen_coord = m_vp * NDC;
		Vec3f screen_coord = proj<3>(_screen_coord / _screen_coord[3]);
		return screen_coord;
	}

	mat<4, 4, float> normalCorrect() const
	{
		return (m_proj * m_cam).invert_transpose();
		 
		// bugs may lives here.., the sign of outputs is not correct.
		/*mat<4, 4, float> m = Matrix::negative_identity();
		if (persp_tag)
			return (m_proj * m_cam * m).invert_transpose();
		else
			return (m_proj * m_cam).invert_transpose();*/
	
	}
	
	//initial
	void computeMatrices()
	{
		m_vp = M_viewport();
		m_proj = M_project();
		m_cam = M_camera();
	}

	BufferParams() {}
};

// draw lines and initiation
void line(Vec2i p1, Vec2i p2, HDC& hdc, TGAColor color)
{
	// Bresenham line drawing
	auto f_xy = [](float x, float y, int x0, int x1, int y0, int y1)
	{
		float f_xy = (float)((y0 - y1) * x) + (float)(x1 - x0) * y + (float)(x0 * y1 - x1 * y0);
		return f_xy;
	};

	int x0 = p1.x;
	int y0 = p1.y;
	int x1 = p2.x;
	int y1 = p2.y;

	//f(x+1,y) = f(x,y) + (y0-y1)
	//f(x+1,y+1) = f(x,y) + (y0-y1) + (x1-x0)
	if (x0 == x1)	//simplest case
	{
		if (y0 > y1)
			std::swap(y0, y1);

		for (int y = y0; y <= y1; y++)
			SetPixel(hdc, x0, y, RGB(color[2], color[1], color[0]));

		return;
	}
	if (y0 == y1)	//simplest case
	{
		if (x0 > x1)
			std::swap(x0, x1);

		for (int x = x0; x <= x1; x++)
			SetPixel(hdc, x, y0, RGB(color[2], color[1], color[0]));

		return;
	}

	//THIS PROCEDURE SHOULD BE FIRST!!
	bool steep = false;
	if (std::abs(x0 - x1) < std::abs(y0 - y1))
	{
		std::swap(x0, y0);
		std::swap(x1, y1);
		steep = true;
	}

	if (x0 > x1)
	{
		std::swap(x0, x1);
		std::swap(y0, y1);
	}

	//f(x+1,y) = f(x,y) + (y0-y1)
	//f(x+1,y+1) = f(x,y) + (y0-y1) + (x1-x0)
	//f(x+1,y-1) = f(x,y) + (y0-y1) - (x1-x0)
	int y_d = y0;
	float fxy;
	int x_delta = x1 - x0;
	int y_delta = y0 - y1;

	if (y0 < y1)
	{
		fxy = f_xy(x0, y_d + 0.5f, x0, x1, y0, y1);
		for (int x = x0; x <= x1; x++)
		{
			if (fxy >= 0)	//notice the "equal", necessary
			{
				if (!steep)
					SetPixel(hdc, x, y_d, RGB(color[2], color[1], color[0]));
				else
					SetPixel(hdc, y_d, x, RGB(color[2], color[1], color[0]));
				fxy = fxy + y_delta;
			}
			else
			{
				y_d++;
				if (!steep)
					SetPixel(hdc, x, y_d, RGB(color[2], color[1], color[0]));
				else
					SetPixel(hdc, y_d, x, RGB(color[2], color[1], color[0]));
				fxy = fxy + y_delta + x_delta;
			}
		}
	}
	else if (y0 > y1)
	{
		fxy = f_xy(x0, y_d - 0.5f, x0, x1, y0, y1);
		for (int x = x0; x <= x1; x++)
		{
			if (fxy <= 0)	//notice the "equal", necessary
			{
				if (!steep)
					SetPixel(hdc, x, y_d, RGB(color[2], color[1], color[0]));
				else
					SetPixel(hdc, y_d, x, RGB(color[2], color[1], color[0]));
				fxy = fxy + y_delta;
			}
			else
			{
				y_d--;
				if (!steep)
					SetPixel(hdc, x, y_d, RGB(color[2], color[1], color[0]));
				else
					SetPixel(hdc, y_d, x, RGB(color[2], color[1], color[0]));
				fxy = fxy + y_delta - x_delta;
			}
		}
	}
}
void solidBackground(HDC& hdc, TGAColor color, const BufferParams& fbp)
{
	for (int y = 0; y < fbp.height; y++)
		for (int x = 0; x < fbp.width; x++)
			SetPixel(hdc, x, y, RGB(color[2], color[1], color[0]));
}
void drawInit(HDC& hdc, const BufferParams& fbp, const BufferParams& sbp)
{
	// axes
	Vec3f world_origin(0.f, 0.f, 0.f);
	Vec3f world_x_axis(1.f, 0.f, 0.f);
	Vec3f world_y_axis(0.f, 1.f, 0.f);
	Vec3f world_z_axis(0.f, 0.f, 1.f);
	Vec2i screen_origin_i = proj<2>(fbp.world2screen(embed<4>(world_origin)));
	Vec2i screen_xi_axis = proj<2>(fbp.world2screen(embed<4>(world_x_axis)));
	Vec2i screen_yi_axis = proj<2>(fbp.world2screen(embed<4>(world_y_axis)));
	Vec2i screen_zi_axis = proj<2>(fbp.world2screen(embed<4>(world_z_axis)));
	line(screen_origin_i, screen_xi_axis, hdc, red);
	line(screen_origin_i, screen_yi_axis, hdc, green);
	line(screen_origin_i, screen_zi_axis, hdc, blue);

	// light direction
	Vec2i screen_center = proj<2>(fbp.world2screen(embed<4>(sbp.center)));
	Vec2i screen_eye = proj<2>(fbp.world2screen(embed<4>(sbp.eye)));
	line(screen_center, screen_eye, hdc, black);
}

// coords transformation for shadowmap
Vec3f screen2shadow(Vec3f fb_Vertex, const BufferParams& sbp, const BufferParams& fbp)
{
	Matrix w2fb = fbp.m_vp * fbp.m_proj * fbp.m_cam;	// world to framebuffer screen
	Matrix w2sb = sbp.m_vp * sbp.m_proj * sbp.m_cam;	// world to shadowbuffer screen
	Matrix fb2sb = w2sb * (w2fb.invert());	// framebuffer screen to shadowbuffer screen

	Vec4f _sb_Vertex = fb2sb * (embed<4>(fb_Vertex));
	Vec3f sb_Vertex = proj<3>(_sb_Vertex / _sb_Vertex[3]);
	return sb_Vertex;
}

// shaders, for the triangle
struct IShader 
{
	// virtual ~IShader();
	virtual Vec3f vertex(int iface, int nthvert, int obj_index, Model* model,
		const BufferParams& fbp, const BufferParams& sbp, Environment& env) = 0;
	virtual void fragment(Vec3f barycentric_coord, TGAColor& color, int obj_index, Model* model,
		const BufferParams& fbp, const BufferParams& sbp, Environment& env) = 0;
};
struct GouraudShader : public IShader
{
	Vec3f varying_intensity;

	virtual Vec3f vertex(int iface, int nthvert, int obj_index, Model* model,
		const BufferParams& fbp, const BufferParams& sbp, Environment& env)
	{
		varying_intensity[nthvert] = max(0.f, model->normal(iface, nthvert) * (env.getLightDir() * (-1.f)));
		Vec4f gl_Vertex = embed<4>(model->vert(iface, nthvert));
		return fbp.world2screen(gl_Vertex);
	}

	virtual void fragment(Vec3f barycentric_coord, TGAColor& color, int obj_index, Model* model,
		const BufferParams& fbp, const BufferParams& sbp, Environment& env)
	{
		float intensity = varying_intensity * barycentric_coord;
		color = white * intensity;
	}
};
struct FlatcolorShader : public IShader
{
	Vec3f varying_intensity;

	virtual Vec3f vertex(int iface, int nthvert, int obj_index, Model* model, 
		const BufferParams& fbp, const BufferParams& sbp, Environment& env)
	{
		varying_intensity[nthvert] = max(0.f, model->normal(iface, nthvert) * (env.getLightDir() * (-1.f)));
		Vec4f gl_Vertex = embed<4>(model->vert(iface, nthvert));
		return fbp.world2screen(gl_Vertex);
	}

	virtual void fragment(Vec3f bar, TGAColor& color, int obj_index, Model* model, 
		const BufferParams& fbp, const BufferParams& sbp, Environment& env)
	{
		float intensity = varying_intensity * bar;
		if (intensity > .85) intensity = 1;
		else if (intensity > .60) intensity = .80;
		else if (intensity > .45) intensity = .60;
		else if (intensity > .30) intensity = .45;
		else if (intensity > .15) intensity = .30;
		else intensity = 0;
		color = white * intensity;
	}
};
struct PhongShader : public IShader
{
	// PhongShader is designed to be with shadows (or not).
	// bugs exist here

	bool tagShadow = false;
	mat<3, 3, float> varying_sb;	// shadowbuffer
	mat<2, 3, float> varying_uv;
	// mat<4, 4, float> m_normalCorrect = frameBufferParamsPtr->normalCorrect();
	mat<3, 3, float> M_normals;

	// Phong: (vert.)obtain normals->embed->correct->proj to 3f->normalize->(frag.)interpolate->normalize
	Vec3f vertex(int iface, int nthvert, int obj_index, Model* model, 
		const BufferParams& fbp, const BufferParams& sbp, Environment& env)
	{
		varying_uv.set_col(nthvert, model->objectuv(obj_index, iface, nthvert));
		M_normals.set_col(nthvert, proj<3>(fbp.normalCorrect() * (embed<4>(model->objnormal(obj_index, iface, nthvert)))).normalize());

		Vec3f gl_Vertex_3f = model->objvert(obj_index, iface, nthvert);
		Vec4f gl_Vertex = embed<4>(model->objvert(obj_index, iface, nthvert));
		Vec3f fb_Vertex = fbp.world2screen(gl_Vertex);
		if (tagShadow)
			varying_sb.set_col(nthvert, screen2shadow(fb_Vertex, sbp, fbp));

		return fb_Vertex;
	}

	void fragment(Vec3f bar, TGAColor& color, int obj_index, Model* model, 
		const BufferParams& fbp, const BufferParams& sbp, Environment& env)
	{
		Vec2f uv = varying_uv * bar;
		// int p = model->specular(uv);
		int p = 128;
		Vec3f n = (M_normals * bar).normalize();
		Vec3f l = (env.getLightDir() * (-1.f));
		Vec3f g = (fbp.gaze_dir) * (-1.f);
		Vec3f h = (g + l).normalize();
		float c_spec = 0.9f;
		float c_diff = 0.5f;
		float c_ambi = env.getAmbientIntensity();
		float varying_c_shadow = 1.f;	// coloring weight
		float error = 0.01f;	// solve z-fighting
		float spec = c_spec * pow(n * h, p);
		float diff = c_diff * max(0.f, n * l);
		TGAColor ambient = TGAColor(0, 0, 0, 255) * c_ambi;

		TGAColor c = model->diffuse(obj_index, uv);
		color = c;
		
		if (tagShadow)
		{
			Vec3f sf_coords_3f = varying_sb * bar;
			Vec2i sf_coords = proj<2>(sf_coords_3f);
			if (sf_coords_3f.z < sbp.zbuffer[sf_coords.x][sf_coords.y] - error)
				varying_c_shadow = 0.3f;	// shadowing
		}

		for (int i = 0; i < 3; i++)	//R, G, B
		{
			// 0.6 here is the weight of diffused color
			color[i] = (min(c_ambi * ambient[i] + (c[i] * (diff + spec + 0.6f)), 255)) * varying_c_shadow;
		}
	}

	void setShadow(bool _tagShadow){tagShadow = _tagShadow;}
};
struct DiffuseShader : public IShader
{
	mat<2, 3, float> varying_uv;

	Vec3f vertex(int iface, int nthvert, int obj_index, Model* model, 
		const BufferParams& fbp, const BufferParams& sbp, Environment& env)
	{
		varying_uv.set_col(nthvert, model->objectuv(obj_index, iface, nthvert));

		Vec4f worldVertex = embed<4>(model->objvert(obj_index, iface, nthvert));
		Vec3f cvvVertex = fbp.world2CVV(worldVertex);
		return cvvVertex;
	}

	void fragment(Vec3f bar, TGAColor& color, int obj_index, Model* model, 
		const BufferParams& fbp, const BufferParams& sbp, Environment& env)
	{
		Vec2f uv = varying_uv * bar;
		color = model->diffuse(obj_index, uv);
	}
};
struct DepthShader : public IShader
{
	// render the depth, by shader
	mat<3, 3, float> varying_tri;

	virtual Vec3f vertex(int iface, int nthvert, int obj_index, Model* model, 
		const BufferParams& fbp, const BufferParams& sbp, Environment& env)
	{
		Vec4f gl_Vertex = embed<4>(model->vert(iface, nthvert));
		Vec3f gl_Vertex_3f = sbp.world2screen(gl_Vertex);
		varying_tri.set_col(nthvert, gl_Vertex_3f);
		return gl_Vertex_3f;
	}

	virtual void fragment(Vec3f bar, TGAColor& color, int obj_index, Model* model, 
		const BufferParams& fbp, const BufferParams& sbp, Environment& env)
	{
		Vec3f p = varying_tri * bar;
		color = white * (p.z);
	}
};
struct ZShader : public IShader
{
	// wrote only for zbuffer updating (for ambient occlusion)
	virtual Vec3f vertex(int iface, int nthvert, int obj_index, Model* model, const BufferParams& fbp, const BufferParams& sbp, Environment& env)
	{
		Vec4f gl_Vertex = embed<4>(model->vert(iface, nthvert));
		return fbp.world2screen(gl_Vertex);
	}

	virtual void fragment(Vec3f barycentric_coord, TGAColor& color, int obj_index, Model* model,
		const BufferParams& fbp, const BufferParams& sbp, Environment& env) { color = TGAColor(0, 0, 0);}
};

void drawModelFrame(Model* model, HDC& hdc, const BufferParams& fbp)
{
	for (int i = 0; i < model->nfaces(); i++)
	{
		std::vector<int> face = model->face(i);
		for (int j = 0; j < 3; j++)
		{
			Vec3f world_coord_v0 = model->vert(face[j]);
			Vec3f world_coord_v1 = model->vert(face[(j + 1) % 3]);
			Vec3f screen_coord_v0 = fbp.world2screen(embed<4>(world_coord_v0));
			Vec3f screen_coord_v1 = fbp.world2screen(embed<4>(world_coord_v1));
			Vec2i screen_coord_v0i = proj<2>(screen_coord_v0);
			Vec2i screen_coord_v1i = proj<2>(screen_coord_v1);
			line(screen_coord_v0i, screen_coord_v1i, hdc, white);
		}
	}
}
void clippingWithPlane(std::vector<Vec3f> &inputVertices, const Plane& pl)
{
	std::vector<Vec3f> outputVertices;
	for (int i = 0; i < inputVertices.size(); i++)
	{
		Vec3f head = inputVertices[i];
		Vec3f tail = inputVertices[(i + 1) % inputVertices.size()];
		if (pointPlaneNormalSide(head, pl))outputVertices.push_back(head);

		LineSegment ls = LineSegment(head, tail);
		Vec3f intersection;
		if (lineIntersectPlane(ls, pl, intersection))
			outputVertices.push_back(intersection);

	}

	inputVertices = outputVertices;
}
void clippingStage(std::vector<Vec3f>& inputVertices)
{
	// note: inner normals
	Plane topPlane(Vec3f(1, 1, -1), Vec3f(0, -1, 0));
	Plane bottomPlane(Vec3f(-1, -1, 1), Vec3f(0, 1, 0));
	Plane leftPlane(Vec3f(-1, -1, 1), Vec3f(1, 0, 0));
	Plane rightPlane(Vec3f(1, 1, -1), Vec3f(-1, 0, 0));
	Plane nearPlane(Vec3f(-1, -1, 1), Vec3f(0, 0, -1));
	Plane farPlane(Vec3f(1, 1, -1), Vec3f(0, 0, 1));
	
	// test for extreme perspection
	if (!(pointPlaneNormalSide(inputVertices[0], topPlane)
		&& pointPlaneNormalSide(inputVertices[1], topPlane)
		&& pointPlaneNormalSide(inputVertices[2], topPlane)
		&& pointPlaneNormalSide(inputVertices[0], bottomPlane)
		&& pointPlaneNormalSide(inputVertices[1], bottomPlane)
		&& pointPlaneNormalSide(inputVertices[2], bottomPlane)
		&& pointPlaneNormalSide(inputVertices[0], leftPlane)
		&& pointPlaneNormalSide(inputVertices[1], leftPlane)
		&& pointPlaneNormalSide(inputVertices[2], leftPlane)
		&& pointPlaneNormalSide(inputVertices[0], rightPlane)
		&& pointPlaneNormalSide(inputVertices[1], rightPlane)
		&& pointPlaneNormalSide(inputVertices[2], rightPlane)
		&& pointPlaneNormalSide(inputVertices[0], nearPlane)
		&& pointPlaneNormalSide(inputVertices[1], nearPlane)
		&& pointPlaneNormalSide(inputVertices[2], nearPlane)
		&& pointPlaneNormalSide(inputVertices[0], farPlane)
		&& pointPlaneNormalSide(inputVertices[1], farPlane)
		&& pointPlaneNormalSide(inputVertices[2], farPlane))
		)return;

	clippingWithPlane(inputVertices, topPlane);
	clippingWithPlane(inputVertices, bottomPlane);
	clippingWithPlane(inputVertices, leftPlane);
	clippingWithPlane(inputVertices, rightPlane);
	clippingWithPlane(inputVertices, nearPlane);
	clippingWithPlane(inputVertices, farPlane);
}
void triangle(IShader& shader, Model* model, int obj_index, Vec3f (&screen_coords)[3], HDC& hdc,
	BufferParams& bp, BufferParams& fbp, BufferParams& sbp, Environment& env)
{
	auto barycentric_for_triangle = [](Vec2i floor, Vec2i(&p)[3], Vec3f& barycentric, Vec3f& delta_barycentric_x, Vec3f& delta_barycentric_y)
	{
		auto f_xy = [](float x, float y, int x0, int x1, int y0, int y1)
		{
			float f_xy = (float)((y0 - y1) * x) + (float)(x1 - x0) * y + (float)(x0 * y1 - x1 * y0);
			return f_xy;
		};

		// compute bary of the floor point and deltas of bary
		float f12_x0_y0 = f_xy(p[0].x, p[0].y, p[1].x, p[2].x, p[1].y, p[2].y);
		float f20_x1_y1 = f_xy(p[1].x, p[1].y, p[2].x, p[0].x, p[2].y, p[0].y);
		float f01_x2_y2 = f_xy(p[2].x, p[2].y, p[0].x, p[1].x, p[0].y, p[1].y);

		//initialize alpha, beta and gamma
		barycentric[0] = f_xy(floor.x, floor.y, p[1].x, p[2].x, p[1].y, p[2].y) / f12_x0_y0;
		barycentric[1] = f_xy(floor.x, floor.y, p[2].x, p[0].x, p[2].y, p[0].y) / f20_x1_y1;
		barycentric[2] = f_xy(floor.x, floor.y, p[0].x, p[1].x, p[0].y, p[1].y) / f01_x2_y2;

		//the differences for alpha, beta and gamma are constant, when x increments 
		delta_barycentric_x[0] = (p[1].y - p[2].y) / f12_x0_y0;
		delta_barycentric_x[1] = (p[2].y - p[0].y) / f20_x1_y1;
		delta_barycentric_x[2] = (p[0].y - p[1].y) / f01_x2_y2;
		//the differences for alpha, beta and gamma are constant, when y increments 
		delta_barycentric_y[0] = (p[2].x - p[1].x) / f12_x0_y0;
		delta_barycentric_y[1] = (p[0].x - p[2].x) / f20_x1_y1;
		delta_barycentric_y[2] = (p[1].x - p[0].x) / f01_x2_y2;
	};
	auto bary_xincre = [](Vec3f& barycentric_coord, Vec3f& delta_barycentric_x)
	{
		barycentric_coord[0] += delta_barycentric_x[0];
		barycentric_coord[1] += delta_barycentric_x[1];
		barycentric_coord[2] += delta_barycentric_x[2];
	};
	auto bary_changerow = [](Vec3f& barycentric_coord, Vec3f& delta_barycentric_x, Vec3f& delta_barycentric_y, int x_range)
	{
		barycentric_coord[0] = barycentric_coord[0] - x_range * delta_barycentric_x[0] + delta_barycentric_y[0];
		barycentric_coord[1] = barycentric_coord[1] - x_range * delta_barycentric_x[1] + delta_barycentric_y[1];
		barycentric_coord[2] = barycentric_coord[2] - x_range * delta_barycentric_x[2] + delta_barycentric_y[2];
	};

	Vec3f screen_coords_z(screen_coords[0].z, screen_coords[1].z, screen_coords[2].z);
	Vec2i p[3] = { proj<2>(screen_coords[0]), proj<2>(screen_coords[1]), proj<2>(screen_coords[2]) };

	// degenerate triangles
	if (p[0].y == p[1].y && p[0].y == p[2].y) return;
	if (p[0].x == p[1].x && p[0].x == p[2].x) return;

	const int x_floor = min(min(p[0].x, p[1].x), p[2].x);
	const int x_ceiling = max(max(p[0].x, p[1].x), p[2].x);
	const int y_floor = min(min(p[0].y, p[1].y), p[2].y);
	const int y_ceiling = max(max(p[0].y, p[1].y), p[2].y);

	// throw triangles out of screen
	// if (x_floor < 0 || x_ceiling >= bp.width || y_floor < 0 || y_ceiling >= bp.height)return;

	Vec3f barycentric_coord;	//alpha, beta, gamma
	Vec3f delta_barycentric_x;
	Vec3f delta_barycentric_y;

	barycentric_for_triangle(Vec2i(x_floor, y_floor), p, barycentric_coord, delta_barycentric_x, delta_barycentric_y);

	float e = -0.01f;

	// fragment
	for (int y = y_floor; y <= y_ceiling; y++)
	{
		for (int x = x_floor; x <= x_ceiling; x++)
		{
			if (x >= 0 && x < bp.width && y >= 0 && y < bp.height)	//naive clipping
			{
				if (barycentric_coord[0] >= e && barycentric_coord[1] >= e && barycentric_coord[2] >= e)
				{
					float z = barycentric_coord * screen_coords_z;

					if (z > bp.zbuffer[x][y])
					{
						bp.zbuffer[x][y] = z;	// better do it first
						TGAColor color;
						shader.fragment(barycentric_coord, color, obj_index, model, fbp, sbp, env);
						SetPixel(hdc, x, y, RGB(color[2], color[1], color[0]));
					}
				}
			}
			bary_xincre(barycentric_coord, delta_barycentric_x);
		}
		bary_changerow(barycentric_coord, delta_barycentric_x, delta_barycentric_y, x_ceiling - x_floor + 1);
	}
}
void drawModel(IShader& shader, Model* model, HDC& hdc,
	BufferParams& bp, BufferParams& fbp, BufferParams& sbp, Environment& env)
{
	for (int iobject = 0; iobject < model->nobjects(); iobject++)
	{
		for (int iface = 0; iface < model->objectnfaces(iobject); iface++)
		{	
			Vec3f NDCs[3];
			std::vector<Vec3f> inputVertices;
			for (int nthvert = 0; nthvert < 3; nthvert++)
			{
				NDCs[nthvert] = shader.vertex(iface, nthvert, iobject, model, fbp, sbp, env);
				inputVertices.push_back(NDCs[nthvert]);
			}

			// clipping
			clippingStage(inputVertices);
			std::cout << "size: " << inputVertices.size() << std::endl;
			if (inputVertices.size() == 0) continue;

			for (int i = 0; i < inputVertices.size(); i++)
				inputVertices[i] = bp.CVV2screen(embed<4>(inputVertices[i]));

			// triangles assembly
			for (int i = 0; i < inputVertices.size() - 2; i++)
			{
				int idx0 = 0;
				int idx1 = i + 1;
				int idx2 = i + 2;
				Vec3f screen_coords[3] = { inputVertices[idx0] ,inputVertices[idx1] ,inputVertices[idx2] };
				triangle(shader, model, iobject, screen_coords, hdc, bp, fbp, sbp, env);
			}
		}
	}
}

// draw buffer or sth..
void drawBuffer(BufferParams& fbp)
{
	// draw present buffer
	TGAImage buffer(fbp.width, fbp.height, TGAImage::RGB);
	for (int x = 0; x < fbp.width; x++)
		for (int y = 0; y < fbp.height; y++)
			buffer.set(x, y, white * fbp.zbuffer[x][y]);
	buffer.flip_vertically();
	buffer.write_tga_file("output/buffer.tga");
}
void drawAmbientOcclusion(BufferParams& fbp)
{
	// draw present buffer
	TGAImage ambientOcclusion(fbp.width, fbp.height, TGAImage::RGB);

	auto max_elevation_angle = [](BufferParams& fbp, Vec2f pix, Vec2f dir)
	{
		float maxangle = 0;
		for (float step = 0.f; step < 1000.f; step += 1.f)
		{
			Vec2f posi = pix + dir * step;
			if (posi.x >= fbp.width || posi.y >= fbp.height || posi.x < 0 || posi.y < 0) return maxangle;
			float distance = (pix - posi).norm();
			if (distance < 1.f) continue;	// rid off noise
			float elevation = fbp.zbuffer[(int)posi.x][(int)posi.y] - fbp.zbuffer[int(pix.x)][int(pix.y)];
			maxangle = max(maxangle, atanf(elevation / distance));
		}
		return maxangle;
	};
	
	for (int x = 0; x < fbp.width; x++)
		for (int y = 0; y < fbp.height; y++)
		{
			if (fbp.zbuffer[x][y] <= lowest + 1.f) continue;
			float total = 0;
			for (float a = 0; a < PI * 2 - 1e-4; a += PI / 4)
				total += PI / 2 - max_elevation_angle(fbp, Vec2f(x, y), Vec2f(cos(a), sin(a)));

			total /= PI * 4;	// total = total /8 /(PI/2)
			total = pow(total, 100.f);
			ambientOcclusion.set(x, y, white * total);
		}
	
	ambientOcclusion.flip_vertically();
	ambientOcclusion.write_tga_file("output/ambientOcclusion.tga");
}






