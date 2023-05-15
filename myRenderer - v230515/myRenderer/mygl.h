#include <iostream>
#include "tgaimage.h"
#include <vector>
#include <cmath>
#include "model.h"
#include "geometry.h"
#include <limits>

const TGAColor black = TGAColor(0, 0, 0, 255);
const TGAColor white = TGAColor(255, 255, 255, 255);
const TGAColor red = TGAColor(255, 0, 0, 255);
const TGAColor green = TGAColor(0, 255, 0, 255);
const TGAColor blue = TGAColor(0, 0, 255, 255);
const TGAColor light_blue = TGAColor(189, 246, 255, 255);
const TGAColor gray = TGAColor(128, 128, 128, 255);
const TGAColor light_green = TGAColor(133, 214, 183, 255);

struct Environment
{
private:
	// Vec3f light_pos;
	Vec3f light_dir;

public:
	void setLightDir(Vec3f _light_dir)
	{
		light_dir = _light_dir;
	}

	Vec3f getLightDir()
	{
		return light_dir;
	}
};
struct _BufferParams
{
	// screen
	const int width;
	const int height;

	//zbuffer
	float** zbuffer;
	virtual void init_zbuffer() = 0;
	virtual void del_zbuffer() = 0;

	// camera box
	const int left;
	const int right;
	const int bottom;
	const int top;
	const int near;
	const int far;

	// global params
	Vec3f t;	// top pointer of camera
	Vec3f center;
	Vec3f eye;
	Vec3f gaze_dir;
	virtual void lookAt(Vec3f _center, Vec3f _eye) = 0;
	virtual mat<4, 4, float> M_viewport(int w, int h) = 0;
	virtual mat<4, 4, float> M_orthoProj(int l, int r, int b, int t, int n, int f) = 0;
	virtual mat<4, 4, float> M_persProj(int l, int r, int b, int t, int n, int f) = 0;
	virtual mat<4, 4, float> M_camera(Vec3f center, Vec3f t, Vec3f eye) = 0;

	virtual ~_BufferParams();
};
struct BufferParams
{
public:

	// screen
	const int width = 720;
	const int height = 1280;

	// zbuffer
	float** zbuffer;
	void init_zbuffer()
	{
		// note here, rotated by 90 degrees
		zbuffer = new float* [width];
		for (int x = 0; x < width; x++)
			zbuffer[x] = new float[height];

		for (int x = 0; x < width; x++)
			for (int y = 0; y < height; y++)
				zbuffer[x][y] = std::numeric_limits<float>::lowest();
	}
	void del_zbuffer()
	{
		for (int x = 0; x < width; x++)
			delete[] zbuffer[x];

		delete[] zbuffer;
	}

	// camera box
	const int left = -width / 80;
	const int right = width / 80;
	const int bottom = -height / 80;
	const int top = height / 80;
	const int near = -8;
	const int far = -64;

	// perspective params: true -> persp; false -> ortho
	bool persp_tag;
	void setPerspTag(bool _tag)
	{
		persp_tag = _tag;
	}

	// global params
	Vec3f t = Vec3f(0, 1, 0);	// top pointer of camera
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
		m[0][3] = width / 2.f;
		m[1][3] = height / 2.f;

		return m;
	};
	mat<4, 4, float> M_project()
	{
		mat<4, 4, float> m = Matrix::identity();
		if (persp_tag)	// perspective
		{
			// bugs may lives here.., the sign of outputs doesn't matter!?
			m[0][0] = (float)(2 * near) / (float)(right - left);
			m[0][2] = (float)(left + right) / (float)(left - right);
			m[1][1] = (float)(2 * near) / (float)(top - bottom);
			m[1][2] = (float)(bottom + top) / (float)(bottom - top);
			m[2][2] = (float)(far + near) / (float)(near - far);
			m[2][3] = (float)(2 * far * near) / (float)(far - near);
			m[3][2] = 1.f;
			m[3][3] = 0.f;

			m = m * Matrix::negative_identity();


			// perspective matrix supported by chat-gpt
			//m[0][0] = (float)(2 * near) / (float)(right - left);
			//m[0][2] = (float)(left + right) / (float)(right - left);
			//m[1][1] = (float)(2 * near) / (float)(top - bottom);
			//m[1][2] = (float)(bottom + top) / (float)(top - bottom);
			//m[2][2] = (float)(far + near) / (float)(near - far);
			//m[2][3] = (float)(2 * far * near) / (float)(near - far);
			//m[3][2] = -1.f;
			//m[3][3] = 0.f;
		}
		else	// orthogonal
		{
			m[0][0] = 2.f / (right - left);
			m[1][1] = 2.f / (top - bottom);
			m[2][2] = 2.f / (near - far);
			m[0][3] = (float)(right + left) / (float)(left - right);
			m[1][3] = (float)(top + bottom) / (float)(bottom - top);
			m[2][3] = (float)(near + far) / (float)(far - near);
		}
		return m;
	};
	mat<4, 4, float> M_camera()
	{
		// t always points to up
		// center is the gazed point
		Vec3f w = gaze_dir * (-1.f);
		Vec3f u = cross(t, w).normalize();
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
	Vec3f world2screen(Vec4f world_coord)
	{
		Vec4f _screen_coord = (m_vp * m_proj * m_cam) * world_coord;
		Vec3f screen_coord = proj<3>(_screen_coord / _screen_coord[3]);
		return screen_coord;
	}

	mat<4, 4, float> normalCorrect()
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
	void init_Matrices()
	{
		m_vp = M_viewport();
		m_proj = M_project();
		m_cam = M_camera();
	}

	~BufferParams()
	{
		setPerspTag(false);
	}
};
Environment* env = new Environment();
BufferParams* shadowBufferParamsPtr = new BufferParams();
BufferParams* frameBufferParamsPtr = new BufferParams();

Vec3f screen2shadow(Vec3f fb_Vertex, BufferParams* sbPP, BufferParams* fbPP)
{
	Matrix w2fb = fbPP->m_vp * fbPP->m_proj * fbPP->m_cam;	// world to framebuffer screen
	Matrix w2sb = sbPP->m_vp * sbPP->m_proj * sbPP->m_cam;	// world to shadowbuffer screen
	Matrix fb2sb = w2sb * (w2fb.invert());	// framebuffer screen to shadowbuffer screen

	Vec4f _sb_Vertex = fb2sb * (embed<4>(fb_Vertex));
	Vec3f sb_Vertex = proj<3>(_sb_Vertex / _sb_Vertex[3]);
	return sb_Vertex;
}

Model* model = NULL;
const void init_model(const char* filename)
{
	model = new Model(filename);
}

// draw lines and initiation
const void solidBackground(TGAImage& image, TGAColor color)
{
	for (int y = 0; y < frameBufferParamsPtr->height; y++)
		for (int x = 0; x < frameBufferParamsPtr->width; x++)
		{
			image.set(x, y, color);
		}
}
const void line(Vec2i p1, Vec2i p2, TGAImage& image, TGAColor color)
{
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
			image.set(x0, y, color);
		return;
	}
	if (y0 == y1)	//simplest case
	{
		if (x0 > x1)
			std::swap(x0, x1);

		for (int x = x0; x <= x1; x++)
		{
			image.set(x, y0, color);
		}
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
					image.set(x, y_d, color);
				else
					image.set(y_d, x, color);
				fxy = fxy + y_delta;
			}
			else
			{
				y_d++;
				if (!steep)
					image.set(x, y_d, color);
				else
					image.set(y_d, x, color);
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
					image.set(x, y_d, color);
				else
					image.set(y_d, x, color);
				fxy = fxy + y_delta;
			}
			else
			{
				y_d--;
				if (!steep)
					image.set(x, y_d, color);
				else
					image.set(y_d, x, color);
				fxy = fxy + y_delta - x_delta;
			}
		}
	}
}
const void drawInit(TGAImage& image)
{
	// axes
	Vec3f world_origin(0.f, 0.f, 0.f);
	Vec3f world_x_axis(1.f, 0.f, 0.f);
	Vec3f world_y_axis(0.f, 1.f, 0.f);
	Vec3f world_z_axis(0.f, 0.f, 1.f);
	Vec2i screen_origin_i = proj<2>(frameBufferParamsPtr->world2screen(embed<4>(world_origin)));
	Vec2i screen_xi_axis = proj<2>(frameBufferParamsPtr->world2screen(embed<4>(world_x_axis)));
	Vec2i screen_yi_axis = proj<2>(frameBufferParamsPtr->world2screen(embed<4>(world_y_axis)));
	Vec2i screen_zi_axis = proj<2>(frameBufferParamsPtr->world2screen(embed<4>(world_z_axis)));
	line(screen_origin_i, screen_xi_axis, image, red);
	line(screen_origin_i, screen_yi_axis, image, green);
	line(screen_origin_i, screen_zi_axis, image, blue);

	// light direction
	Vec2i screen_center = proj<2>(frameBufferParamsPtr->world2screen(embed<4>(shadowBufferParamsPtr->center)));
	Vec2i screen_eye = proj<2>(frameBufferParamsPtr->world2screen(embed<4>(shadowBufferParamsPtr->eye)));
	line(screen_center, screen_eye, image, black);
}

// shaders, for the triangle
struct IShader 
{
	// virtual ~IShader();
	virtual Vec3f vertex(int iface, int nthvert) = 0;
	virtual void fragment(Vec3f barycentric_coord, TGAColor& color) = 0;
};
struct GouraudShader : public IShader
{
	Vec3f varying_intensity;

	virtual Vec3f vertex(int iface, int nthvert) 
	{
		varying_intensity[nthvert] = std::max(0.f, model->normal(iface, nthvert) * (env->getLightDir() * (-1.f)));
		Vec4f gl_Vertex = embed<4>(model->vert(iface, nthvert));
		return frameBufferParamsPtr->world2screen(gl_Vertex);
	}

	virtual void fragment(Vec3f barycentric_coord, TGAColor& color) 
	{
		float intensity = varying_intensity * barycentric_coord;
		color = white * intensity;
	}
};
struct FlatcolorShader : public IShader
{
	Vec3f varying_intensity;

	virtual Vec3f vertex(int iface, int nthvert)
	{
		varying_intensity[nthvert] = std::max(0.f, model->normal(iface, nthvert) * (env->getLightDir() * (-1.f)));
		Vec4f gl_Vertex = embed<4>(model->vert(iface, nthvert));
		return frameBufferParamsPtr->world2screen(gl_Vertex);
	}

	virtual void fragment(Vec3f bar, TGAColor& color)
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
struct TexShader : public IShader
{
	// naively covering a tex sheet
	Vec3f varying_intensity;
	mat<2, 3, float> varying_uv;

	virtual Vec3f vertex(int iface, int nthvert)
	{
		varying_uv.set_col(nthvert, model->uv(iface, nthvert));
		varying_intensity[nthvert] = std::max(0.f, model->normal(iface, nthvert) * (env->getLightDir() * (-1.f)));
		Vec4f gl_Vertex = embed<4>(model->vert(iface, nthvert));
		return frameBufferParamsPtr->world2screen(gl_Vertex);
	}

	virtual void fragment(Vec3f bar, TGAColor& color)
	{
		// delete
		float intensity = varying_intensity * bar;
		Vec2f uv = varying_uv * bar;
		color = model->diffuse(uv) * intensity;
	}
};
struct PhongShader : public IShader
{
	mat<2, 3, float> varying_uv;
	mat<3, 3, float> varying_sb;	// shadowbuffer
	mat<4, 4, float> m_normalCorrect = frameBufferParamsPtr->normalCorrect();
	mat<3, 3, float> M_normals;

	// Phong: (vert.)obtain normals->embed->correct->proj to 3f->normalize->(frag.)interpolate->normalize
	virtual Vec3f vertex(int iface, int nthvert)
	{
		varying_uv.set_col(nthvert, model->uv(iface, nthvert));
		M_normals.set_col(nthvert, proj<3>(m_normalCorrect * (embed<4>(model->normal(iface, nthvert)))).normalize());

		Vec3f gl_Vertex_3f = model->vert(iface, nthvert);
		Vec4f gl_Vertex = embed<4>(model->vert(iface, nthvert));
		Vec3f fb_Vertex = frameBufferParamsPtr->world2screen(gl_Vertex);
		varying_sb.set_col(nthvert, screen2shadow(fb_Vertex, shadowBufferParamsPtr, frameBufferParamsPtr));

		return fb_Vertex;
	}

	virtual void fragment(Vec3f bar, TGAColor& color)
	{
		Vec2f uv = varying_uv * bar;
		// int p = model->specular(uv);
		int p = 128;
		Vec3f n = (M_normals * bar).normalize();
		Vec3f l = (env->getLightDir() * (-1.f));
		Vec3f g = (frameBufferParamsPtr->gaze_dir) * (-1.f);
		Vec3f h = (g + l).normalize();
		float c_spec = 0.9f;
		float c_diff = 0.5f;
		float c_ambi = 1.f;
		float varying_c_shadow = 1.f;
		float error = 0.01f;	// solve z-fighting
		float spec = c_spec * pow(n * h, p);
		float diff = c_diff * std::max(0.f, n * l);
		TGAColor ambient = TGAColor(5, 5, 5, 255) * c_ambi;

		// TGAColor c = model->diffuse(uv);
		TGAColor c = white;
		color = c;
		
		Vec3f sf_coords_3f = varying_sb * bar;
		Vec2i sf_coords = proj<2>(sf_coords_3f);

		if (sf_coords_3f.z < shadowBufferParamsPtr->zbuffer[sf_coords.x][sf_coords.y] - error)
		{
			varying_c_shadow = 0.0f;	// shadowing
		}
		else
		{
			varying_c_shadow = 1.f;
		}
		for (int i = 0; i < 3; i++)	//R, G, B
			color[i] = (std::min<float>(c_ambi * ambient[i] + (c[i] * (diff + spec)), 255)) * varying_c_shadow;
	}
};
struct DepthShader : public IShader
{
	mat<3, 3, float> varying_tri;

	DepthShader() :varying_tri() {}

	virtual Vec3f vertex(int iface, int nthvert)
	{
		Vec4f gl_Vertex = embed<4>(model->vert(iface, nthvert));
		Vec3f gl_Vertex_3f = shadowBufferParamsPtr->world2screen(gl_Vertex);
		varying_tri.set_col(nthvert, gl_Vertex_3f);
		return gl_Vertex_3f;
	}

	virtual void fragment(Vec3f bar, TGAColor& color)
	{
		Vec3f p = varying_tri * bar;
		color = white * (p.z);
	}
};

// draw triangles
void triangle(IShader& shader, Vec3f (&screen_coords)[3], TGAImage& image, BufferParams* buffeParamsPtr)
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

	Vec3f barycentric_coord;	//alpha, beta, gamma
	Vec3f delta_barycentric_x;
	Vec3f delta_barycentric_y;

	const int x_floor = std::min({ p[0].x, p[1].x, p[2].x });
	const int x_ceiling = std::max({ p[0].x, p[1].x, p[2].x });
	const int y_floor = std::min({ p[0].y, p[1].y, p[2].y });
	const int y_ceiling = std::max({ p[0].y, p[1].y, p[2].y });

	barycentric_for_triangle(Vec2i(x_floor, y_floor), p, barycentric_coord, delta_barycentric_x, delta_barycentric_y);

	float e = -0.01f;

	// fragment
	for (int y = y_floor; y <= y_ceiling; y++)
	{
		for (int x = x_floor; x <= x_ceiling; x++)
		{
			if (x >= 0 && x < buffeParamsPtr->width && y >= 0 && y < buffeParamsPtr->height)	//naive clipping
			{
				if (barycentric_coord[0] >= e && barycentric_coord[1] >= e && barycentric_coord[2] >= e)
				{
					float z = barycentric_coord * screen_coords_z;

					if (z > buffeParamsPtr->zbuffer[x][y])
					{
						TGAColor color;
						shader.fragment(barycentric_coord, color);
						image.set(x, y, color);
						buffeParamsPtr->zbuffer[x][y] = z;
					}
				}
			}
			bary_xincre(barycentric_coord, delta_barycentric_x);
		}
		bary_changerow(barycentric_coord, delta_barycentric_x, delta_barycentric_y, x_ceiling - x_floor + 1);
	}
}
void drawModelFrame(Model* model, TGAImage& image)
{
	for (int i = 0; i < model->nfaces(); i++)
	{
		std::vector<int> face = model->face(i);

		for (int j = 0; j < 3; j++)
		{
			Vec3f world_coord_v0 = model->vert(face[j]);
			Vec3f world_coord_v1 = model->vert(face[(j + 1) % 3]);
			Vec3f screen_coord_v0 = frameBufferParamsPtr->world2screen(embed<4>(world_coord_v0));
			Vec3f screen_coord_v1 = frameBufferParamsPtr->world2screen(embed<4>(world_coord_v1));
			Vec2i screen_coord_v0i = proj<2>(screen_coord_v0);
			Vec2i screen_coord_v1i = proj<2>(screen_coord_v1);
			line(screen_coord_v0i, screen_coord_v1i, image, white);
		}
	}
}
void drawModel(IShader& shader, Model* model, TGAImage& image, BufferParams* buffeParamsPtr)
{
	for (int iface = 0; iface < model->nfaces(); iface++)
	{
		std::vector<int> face = model->face(iface);
		Vec3f screen_coords[3];

		for (int nthvert = 0; nthvert < 3; nthvert++)
		{
			screen_coords[nthvert] = shader.vertex(iface, nthvert);
		}

		triangle(shader, screen_coords, image, buffeParamsPtr);

	}
}
