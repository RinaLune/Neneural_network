import pyglet
from pyglet import app, gl, graphics, clock
from pyglet.window import Window, key
import numpy as np
import random
import math

# R and n is changeable
R = 30
n = 6

w = 3 * R 

vx = [R*math.cos(i*2*math.pi/n) for i in range (n)]
vy = [R*math.sin(i*2*math.pi/n) for i in range (n)]
vz = [R, -R]

P0 = [-2*R, 0, 0, 0]
P1 = [ 2*R, 0, 0, 0]
C0 = [1, 1, 0, 1]
C1 = [0, 0, 1, 1]

rot_x, rot_y, rot_z = 0, 0, 0
rot_x0, rot_y0, rot_z0 = 0, 0, 0
rot_x1, rot_y1, rot_z1 = 0, 0, 0

flx, fly, flz, fl = False, False, False, False
fl0x, fl0y, fl0z, fl0 = False, False, False, False
fl1x, fl1y, fl1z, fl1 = False, False, False, False

col = [(0.7, 0.308, 0.4975), (1, 0.972, 0.72)]

rot_bx, rot_by, rot_bz = 120, 180, 45

f = 1
k = 5

flm = [True for i in range(8)]
ff = [False for i in range(3)]

def vert(i,j):
    if (j==0):
        mx = vx[i-1]
        my = vy[i-1]
        mz = vz[0]
    elif (j==1):
        mx = vx[i]
        my = vy[i]
        mz = vz[0]
    elif (j==2):
        mx = vx[i]
        my = vy[i]
        mz = vz[1]
    else:
        mx = vx[i-1]
        my = vy[i-1]
        mz = vz[1]
    return mx, my, mz
   
def Normal(M):
    for W in range (n):
        L = 0
        for i in range (3):
            L += (M[W][i])**2
        S = math.sqrt(L)
        M[W] = [M[W][i]/S for i in range (3)]
    return M

def Normalk(M, kbuf):
    for W in range (n):
        for i in range (3):
            M[W][i] = M[W][i]*kbuf
    return M

def Norm(kbuf=0, j=0):
    M1, M2, M3, M4 = [], [], [], []
    M1 = [np.cross([0, 0, vz[1]-vz[0]], [vx[i]-vx[i-1], vy[i]-vy[i-1], 0]) for i in range (n)]
    for i in range (n):
        if j == 1 or j == 2:
            v = [M1[i][0]+M1[(i+1)%n][0], M1[i][1]+M1[(i+1)%n][1], M1[i][2]+M1[(i+1)%n][2]]
        else:
            v = [M1[i-1][0]+M1[i][0], M1[i-1][1]+M1[i][1], M1[i-1][2]+M1[i][2]]
        M2 += [v]
    for i in range (n):
        v1 = [vx[i]-vx[i-1], vy[i]-vy[i-1], 0]
        v2 = [vx[i-2]-vx[i-1], vy[i-2]-vy[i-1], 0]
        if kbuf == 0:
           M4 += [np.cross(v1, v2)]
        else:
           M4 += [np.cross(v2, v1)]
    M3 = [[M2[i][0]+M4[i][0], M2[i][1]+M4[i][1], M2[i][2]+M4[i][2]] for i in range (n)]
    return M1, M2, M3, M4

def drawline(x1, y1, z1, x2, y2, z2):
    graphics.draw(2, gl.GL_LINES,
        ('v3f', (x1, y1, z1, x1+x2, y1+y2, z1+z2)),
        ('c3f', col[0]+col[1]))

def drawpoint(x1, y1, z1):
    gl.glPointSize(5)
    gl.glEnable(gl.GL_POINT_SMOOTH)
    gl.glBegin(gl.GL_POINTS)
    gl.glColor3f(0, 1, 0)
    gl.glVertex3f(x1, y1, z1)
    gl.glEnd()

def drawlines(f=1):
    global k
    for i in range (n):
        for j in range (4):
            mx, my, mz = vert(i,j)
            if j == 0 or j == 1:
                M1, M2, M3, M4 = Norm(0, j)
            else:
                M1, M2, M3, M4 = Norm(1, j)
            f = f%2
            if f == 1:
                if flm[4]:
                    M1 = Normalk(Normal(M1), k)
                    M4 = Normalk(Normal(M4), k)
                drawline(mx, my, mz, M1[i][0], M1[i][1], M1[i][2])
                drawline(mx, my, mz, M4[i][0], M4[i][1], M4[i][2])
            else:
                if flm[4]:
                    M3 = Normalk(Normal(M3), k)
                drawline(mx, my, mz, M3[i][0], M3[i][1], M3[i][2])

def drawO():
    global vx, vy, vz
    
    gl.glEnable(gl.GL_LINE_STIPPLE)
    gl.glLineStipple(2, 0xFBBF)

    graphics.draw(2, gl.GL_LINES,
        ('v3f', (3*w/4, 0, 0, -3*w/4, 0, 0)),
        ('c3f', col[0]+col[1]))
    
    graphics.draw(2, gl.GL_LINES,
        ('v3f', (0, 3*w/4, 0, 0, -3*w/4, 0)),
        ('c3f', col[0]+col[1]))

    graphics.draw(2, gl.GL_LINES,
        ('v3f', (0, 0, 3*w/4, 0, 0, -3*w/4)),
        ('c3f', col[0]+col[1]))
    
    gl.glDisable(gl.GL_LINE_STIPPLE)
     
def drawP():
    global flg
    global k, f
    for i in range (n):
        gl.glBegin(gl.GL_QUADS)
        for j in range(4):
            mx, my, mz = vert(i,j)
            if j==0 or j==1:
                gl.glColor3f(col[0][0], col[0][1], col[0][2])
                M1, M2, M3, M4 = Norm(0, j)
            else:
                gl.glColor3f(col[1][0], col[1][1], col[1][2])
                M1, M2, M3, M4 = Norm(1, j)
            f = f%2
            if f == 1:
                if flm[4]:
                    M1 = Normalk(Normal(M1), k)
                gl.glNormal3f(M1[i][0], M1[i][1], M1[i][2])
            else:
                if flm[4]:
                    M3 = Normalk(Normal(M3), k)
                gl.glNormal3f(M3[i][0], M3[i][1], M3[i][2])
            gl.glVertex3f(mx, my, mz)
        gl.glEnd()
    if flm[0]:
        for i in range (2):
            gl.glBegin(gl.GL_POLYGON)
            for j in range (n):
                if i == 0:
                    gl.glColor3f(col[0][0], col[0][1], col[0][2])
                else:
                    gl.glColor3f(col[1][0], col[1][1], col[1][2])
                M1, M2, M3, M4 = Norm(i, 1)
                f = f%2
                if f == 1:
                    if flm[4]:
                        M4 = Normalk(Normal(M4), k)
                    gl.glNormal3f(M4[j][0], M4[j][1], M4[j][2])
                else:
                    if flm[4]:
                        M3 = Normalk(Normal(M3), k)
                    gl.glNormal3f(M3[j][0], M3[j][1], M3[j][2])
                gl.glVertex3f(vx[j], vy[j], vz[i]) 
            gl.glEnd()

def lightPC(Pbuf, Cbuf, light, color):
    P = (gl.GLfloat * 4)()
    C = (gl.GLfloat * 4)()
    
    for i in range (4):
        P[i] = Pbuf[i]
        C[i] = Cbuf[i]
    gl.glLightfv(light, color, C)
    gl.glLightfv(light, gl.GL_POSITION, P)

def light(a=0):
    global P0, P1, C0, C1
    global rot_x0, rot_y0, rot_z0
    global rot_x1, rot_y1, rot_z1
    global fl0x, fl0y, fl0z, fl0
    global fl1x, fl1y, fl1z, fl1

    fl0, fl0x, fl0y, fl0z, rot_x0, rot_y0, rot_z0 = lightn(P0, C0, gl.GL_LIGHT0, gl.GL_DIFFUSE, fl0, fl0x, fl0y, fl0z, rot_x0, rot_y0, rot_z0)
    fl1, fl1x, fl1y, fl1z, rot_x1, rot_y1, rot_z1 = lightn(P1, C1, gl.GL_LIGHT1, gl.GL_DIFFUSE, fl1, fl1x, fl1y, fl1z, rot_x1, rot_y1, rot_z1)

def lightn(Pbuf, Cbuf, light, color, fl, flx, fly, flz, rot_bufx, rot_bufy, rot_bufz):
    global rot_bx, rot_by, rot_bz
    
    gl.glLoadIdentity()

    gl.glRotatef(rot_bx, 1, 0, 0)
    gl.glRotatef(rot_by, 0, 1, 0)
    gl.glRotatef(rot_bz, 0, 0, 1)

    gl.glRotatef((rot_bufx)%360, 1, 0, 0)
    gl.glRotatef((rot_bufy)%360, 0, 1, 0)
    gl.glRotatef((rot_bufz)%360, 0, 0, 1)

    gl.glPointSize(16)
    gl.glEnable(gl.GL_POINT_SMOOTH)
    gl.glBegin(gl.GL_POINTS)
    gl.glColor3f(Cbuf[0], Cbuf[1], Cbuf[2])
    gl.glVertex3f(Pbuf[0], Pbuf[1], Pbuf[2])
    gl.glEnd()

    lightPC(Pbuf, Cbuf, light, color)

    if fl:
        flx = False
        fly = False
        flz = False

    if flx:
        rot_bufx += 1
    if fly:
        rot_bufy += 1
    if flz:
        rot_bufz += 1
    
    return fl, flx, fly, flz, rot_bufx, rot_bufy, rot_bufz

def draw(a=0):
    global vx, vy, vz
    global flx, fly, flz, fl
    global rot_x, rot_y, rot_z

    gl.glRotatef(rot_x%360, 1, 0, 0)
    gl.glRotatef(rot_y%360, 0, 1, 0)
    gl.glRotatef(rot_z%360, 0, 0, 1)

    if fl:
        flx = False
        fly = False
        flz = False

    if flx:
        rot_x += 1
    if fly:
        rot_y += 1
    if flz:
        rot_z += 1

    drawP()

width, height = int(10 * w), int(10 * w) 
window = Window(visible = True, width = width, height = height,
                resizable = True, caption = 'LR5')
gl.glClearColor(0, 0.6975, 0.93, 1.0)
#gl.glClearColor(1, 1, 1, 1.0)
gl.glClear(gl.GL_COLOR_BUFFER_BIT)
gl.glEnable(gl.GL_LINE_SMOOTH)
gl.glEnable(gl.GL_POLYGON_SMOOTH);
@window.event
def on_draw():
    global vx, vy, vz
    global flx, fly, flz, fl
    global rot_x, rot_y, rot_z
    window.clear()
    gl.glMatrixMode(gl.GL_PROJECTION)
    gl.glLoadIdentity()
    if flm[2]:
        gl.glEnable(gl.GL_DEPTH_TEST)
    else:
        gl.glDisable(gl.GL_DEPTH_TEST)

    if not flm[3]:
        gl.glEnable(gl.GL_CULL_FACE)
        gl.glCullFace(gl.GL_FRONT)
    else:
        gl.glDisable(gl.GL_CULL_FACE)
    
    if not flm[5]:
        gl.glEnable(gl.GL_NORMALIZE)
    else:
        gl.glDisable(gl.GL_NORMALIZE)

    if flm[7]:
        gl.glEnable(gl.GL_COLOR_MATERIAL)
        Cbuf = [1, 1, 1, 1]
        C = (gl.GLfloat * 4)()
        for i in range (4):
            C[i] = Cbuf[i]
        gl.glMaterialfv(gl.GL_FRONT_AND_BACK, gl.GL_DIFFUSE, C)
    else:
        gl.glDisable(gl.GL_COLOR_MATERIAL)

    if flm[6]:
        gl.glEnable(gl.GL_LIGHTING)
        gl.glEnable(gl.GL_LIGHT0)
        gl.glEnable(gl.GL_LIGHT1)
    else:
        gl.glDisable(gl.GL_LIGHTING)

    gl.glOrtho(-w, w, -w, w, -w, w)

    gl.glMatrixMode(gl.GL_MODELVIEW)
    gl.glLoadIdentity()
    
    gl.glRotatef(rot_bx, 1, 0, 0)
    gl.glRotatef(rot_by, 0, 1, 0)
    gl.glRotatef(rot_bz, 0, 0, 1)

    drawO()
    draw()
    if flm[6]: 
        if flm[1]:
            drawlines(f)
        light()

@window.event
def on_key_press(char, modifiers):  
    global flx, fly, flz, fl
    global rot_x, rot_y, rot_z  
    global fl0x, fl0y, fl0z, fl0
    global rot_x0, rot_y0, rot_z0  
    global fl1x, fl1y, fl1z, fl1
    global rot_x1, rot_y1, rot_z1
    global f, k
    
    #Включение/отключение вращения фигуры
    if char == key._1:
        if not ff[0]:
            clock.schedule_interval(draw, 0.01)
            ff[0] = True
        else:
            fl = True
            clock.unschedule(draw)
            ff[0] = False
    #Поворот фигуры относительно оси X
    elif char == key._2:
        flx = True
        fl = False
    #Поворот фигуры относительно оси Y
    elif char == key._3:
        fly = True
        fl = False
    #Поворот фигуры относительно оси Z
    elif char == key._4:
        flz = True
        fl = False
    #Выбор нормалей (от граней/от вершин)
    elif char == key._5:
        f += 1
    #Увелечение длин нормалей
    elif char == key._6:
        k += 1
    #Уменьшение длин нормалей
    elif char == key._7:
        k -= 1
    #Включение/отключение вращения источника1
    elif char == key.Q:
        if not ff[1]:
            clock.schedule_interval(light, 0.01)
            ff[1] = True
        else:
            fl0  = True
            clock.unschedule(light)
            ff[1] = False
    #Поворот источника1 относительно оси X
    elif char == key.W:
        fl0x = True
        fl0 = False
    #Поворот источника1 относительно оси Y
    elif char == key.E:
        fl0y = True
        fl0 = False
    #Поворот источника1 относительно оси Z
    elif char == key.R:
        fl0z = True
        fl0 = False
    #Включение/отключение вращения источника2
    elif char == key.A:
        if not ff[2]:
            clock.schedule_interval(light, 0.01)
            ff[2] = True
        else:
            fl1  = True
            clock.unschedule(light) 
            ff[2] = False
    #Поворот источника2 относительно оси X
    elif char == key.S:
        fl1x = True
        fl1 = False
    #Поворот источника2 относительно оси Y
    elif char == key.D:
        fl1y = True
        fl1 = False
    #Поворот источника2 относительно оси Z
    elif char == key.F:
        fl1z = True
        fl1 = False
    #Отключение/Включение вывода оснований
    elif char == key.Y:
        if flm[0]:
            flm[0] = False
        else:
            flm[0] = True
    #Включение/отключение отображения нормалей
    elif char == key.U:
        if flm[1]:
            flm[1] = False
        else:
            flm[1] = True
    #Включение/отключение теста глубины
    elif char == key.I:
        if flm[2]:
            flm[2] = False
        else:
            flm[2] = True
    #Отключение/включение режима отсечение нелицевых сторон
    elif char == key.O:
        if flm[3]:
            flm[3] = False
        else:
            flm[3] = True
    #Нормализация/отказ от нормализации нормалей при их расчете
    elif char == key.H:
        if flm[4]:
            flm[4] = False
        else:
            flm[4] = True
    #Нормализацию/отказ от нормализации нормалей посредством glEnable(GL_NORMALIZE)
    elif char == key.J:
        if flm[5]:
            flm[5] = False
        else:
            flm[5] = True
    #Отключение/включение режима расчета освещенности
    elif char == key.K:
        if flm[6]:
            flm[6] = False
        else:
            flm[6] = True
    #Отключение/включение компоненты материала GL_DIFFUSE
    elif char == key.L:
        if flm[7]:
            flm[7] = False
        else:
            flm[7] = True
app.run()
