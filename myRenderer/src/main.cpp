#define _CRT_SECURE_NO_DEPRECATE
#include "mygl.h"
#include <iostream>
#include "tgaimage.h"

const bool isShadowing = false;	// draw shadow (only for Phong shading. Set isPhongShading as true.)
const bool isPerspect = true;	// set perspection
const bool isDepth = false;	// draw depth(camera)
const bool isAmbientOcclusion = false;	//draw ambient occlusion (time consuming)
Model* arona = NULL;
Model* yoimiya = NULL;
BufferParams sbp;	// shadow buffer params
BufferParams fbp;	// frame buffer params
Environment env;

void consoleLog()
{
	AllocConsole();
	freopen("conout$", "w", stdout);
	std::cout << "i'm stdout" << std::endl;
	//freopen("conout$", "w", stderr);
	//std::cerr << "i'm cerr" << std::endl;
}
void initialization()
{
	// yoimiya = new Model("model_yoimiya/yoimiya.obj");
	arona = new Model("model_arona/arona.obj");

	if (isShadowing)
	{
		sbp.setPerspTag(false);
		sbp.lookAt_SB(Vec3f(0, 10, 0), env.getLightDir());	
	}

	fbp.setPerspTag(isPerspect);
	fbp.lookAt_FB(Vec3f(0, 10, 0), Vec3f(2, 10, 10));
}
void deinitialization()
{
	// other behaviors
	if (isDepth)drawBuffer(fbp);	// drawing depth
	if (isAmbientOcclusion)drawAmbientOcclusion(fbp);

	// delete yoimiya;
	delete arona;
}

int rendering(HDC &hdc, Model* model)
{
	// render the depth from light at first
	if (isShadowing)
	{
		sbp.computeMatrices();
		sbp.initZbuffer();
		TGAImage depthFromLight(sbp.width, sbp.height, TGAImage::RGB);
		DepthShader depthshader;
		drawModel(depthshader, model, hdc, sbp, fbp, sbp, env);
		depthFromLight.flip_vertically();
		depthFromLight.write_tga_file("output/depthFromLight.tga");
	}

	fbp.computeMatrices();
	fbp.initZbuffer();
	//PhongShader PhongShader;
	//PhongShader.setShadow(isShadowing);	// only for Phong shading
	DiffuseShader diffuseShader;
	drawModel(diffuseShader, model, hdc, fbp, fbp, sbp, env);
	//drawInit(hdc, fbp, sbp);


	if (isShadowing) { sbp.delZbuffer(); }
	fbp.delZbuffer();
	
	return 0;
}

LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
	switch (uMsg)
	{
		case WM_PAINT:
		{
			PAINTSTRUCT ps;
			HDC hdc = BeginPaint(hwnd, &ps);

			// drawing pixels here
			// BG
			SetBkColor(hdc, RGB(255, 192, 203));	// pink
			ExtTextOut(hdc, 0, 0, ETO_OPAQUE, &ps.rcPaint, NULL, 0, NULL);

			rendering(hdc, arona);

			EndPaint(hwnd, &ps);
			return 0;
		}
		case WM_LBUTTONDOWN:
		{

			break;
		}		
		case WM_CLOSE:
		{
			PostQuitMessage(0);
			break;
		}
		default:
		{
			return DefWindowProc(hwnd, uMsg, wParam, lParam);
		}
	}
	return 0;
}

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow)
{
	WNDCLASSEX wc = { sizeof(WNDCLASSEX), CS_CLASSDC, WindowProc, 0L, 0L, GetModuleHandle(NULL), NULL, NULL, NULL, NULL, L"Renderer", NULL };
	RegisterClassEx(&wc);

	HWND hwnd = CreateWindow(wc.lpszClassName, L"Renderer", WS_OVERLAPPEDWINDOW, 100, 100, 1280, 720, NULL, NULL, wc.hInstance, NULL);

	if (hwnd == NULL)
	{
		return 0;
	}

	consoleLog();

	// initialization
	initialization();

	// set windows and draw
	ShowWindow(hwnd, nCmdShow);
	UpdateWindow(hwnd);

	// deinitialization
	deinitialization();

	MSG msg;
	ZeroMemory(&msg, sizeof(msg));

	while (msg.message != WM_QUIT)
	{
		if (PeekMessage(&msg, NULL, 0U, 0U, PM_REMOVE))
		{
			TranslateMessage(&msg);
			DispatchMessage(&msg);
		}
	}

	UnregisterClass(wc.lpszClassName, wc.hInstance);

	return static_cast<int>(msg.wParam);
}
