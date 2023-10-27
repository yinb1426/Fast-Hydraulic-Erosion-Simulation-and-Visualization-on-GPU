import taichi as ti

'''
class Particle:
    def __init__(self, _index: int, _posX: ti.f32, _posY: ti.f32):
        self.index = _index
        self.pos = ti.Vector([_posX, _posY])
        self.velocity = ti.Vector([0.0, 0.0])
        self.volume = 1.0
        self.sediment = 0.0
'''

'''
# 与原程序存在的差别（2023.10.26）：
#   1. 原程序是每一个循环只产生一个粒子，一个粒子在volume<0.01时结束该轮侵蚀
#   2. 该程序在每个格点都产生一个粒子，共sizeX * sizeY个粒子同时开始侵蚀
#   3. 该程序侵蚀速度更快，在deltaT设置较大时会快速将地形侵蚀干净
#   4. (重点)原程序并不是完全并行的，可能不能直接按并行程序去写
'''

# 目前存在的问题：
# 1. 代码检查(应该是没有问题的)
# 2. 参数设置，目前结果OK，但会出现几个白点，结果不稳定

ti.init(arch=ti.gpu)

sizeX = 512
sizeY = 512
deltaT = 0.5
density = 1.0
friction = 0.05
KDeposition = 0.1
KEvaporation = 0.001
minVolume = 0.01

heightData = []

terrainHeight = ti.field(dtype=ti.f32, shape=(sizeX, sizeY))
terrainNormal = ti.Vector.field(3, dtype=ti.f32, shape=(sizeX, sizeY))
terrainHeightMap = ti.Vector.field(3, dtype=ti.f32, shape=(sizeX, sizeY))

# particlePosition = ti.Vector.field(2, dtype=ti.f32, shape=(sizeX, sizeY))
# particleVelocity = ti.Vector.field(2, dtype=ti.f32, shape=(sizeX, sizeY))
# particleVolume = ti.field(dtype=ti.f32, shape=(sizeX, sizeY))
# particleSediment = ti.field(dtype=ti.f32, shape=(sizeX, sizeY))


# 初始化地面高度
def GenerateTerrainHeight():
    for x in range(sizeX):
        print(x)
        for y in range(sizeY):
            terrainHeight[x, y] = heightData[x][y]

'''
# 初始化粒子
@ti.kernel
def InitParticles():
    for x, y in particlePosition:
        particlePosition[x, y] = ti.Vector([x, y])
        particleVelocity[x, y] = ti.Vector([0.0, 0.0])
        particleVolume[x, y] = 1.0
        particleSediment[x, y] = 0.0
'''

# 计算地表法线
@ti.kernel
def CalculateTerrainNormal():
    for x, y in terrainHeight:
        deltaHeightX = terrainHeight[x+1, y] - terrainHeight[x-1, y]
        deltaHeightY = terrainHeight[x, y+1] - terrainHeight[x, y-1]
        R2L = ti.Vector([2.0, 0.0, deltaHeightX])
        B2T = ti.Vector([0.0, 2.0, deltaHeightY])
        normal0 = R2L.cross(B2T)
        normal = normal0 / normal0.norm()
        terrainNormal[x, y] = normal

'''
# 侵蚀过程(粒子移动 -> 边界检测 -> 沉积过程)
@ti.kernel
def Erosion():
    # for x, y in particlePosition:
    for x in range(sizeX):
        for y in range(sizeY):
            if(particleVolume[x, y] < 0.01):
                particlePosition[x, y] = ti.Vector([x, y])
                particleVelocity[x, y] = ti.Vector([0.0, 0.0])
                particleVolume[x, y] = 1.0
                particleSediment[x, y] = 0.0
            else:
                iPos = ti.Vector([particlePosition[x, y][0], particlePosition[x, y][1]])
                normal = terrainNormal[x, y]
                newVelocity = deltaT * ti.Vector([normal[0], normal[1]]) / (particleVolume[x, y] * density)
                particleVelocity[x, y] += newVelocity
                particlePosition[x, y] += deltaT * newVelocity
                particleVelocity[x, y] *= (1.0 - deltaT * friction)

                if(particlePosition[x, y][0] < 0 or particlePosition[x, y][0] > sizeX or 
                    particlePosition[x, y][1] < 0 or particlePosition[x, y][1] > sizeY):
                    particlePosition[x, y] = ti.Vector([x, y])
                    particleVelocity[x, y] = ti.Vector([0.0, 0.0])
                    particleVolume[x, y] = 1.0
                    particleSediment[x, y] = 0.0
                else:
                    c_eq = max(0.0, particleVolume[x, y] * particleVelocity[x, y].norm() * (terrainHeight[int(iPos[0]), int(iPos[1])] - terrainHeight[int(particlePosition[x, y][0]), int(particlePosition[x, y][1])]))
                    cDiff = c_eq - particleSediment[x, y]
                    particleSediment[x, y] += deltaT * KDeposition * cDiff
                    terrainHeight[x, y] -= deltaT * particleVolume[x, y] * KDeposition * cDiff
                    particleVolume[x, y] *= (1.0 - deltaT * KEvaporation)
'''
# 侵蚀过程(粒子移动 -> 边界检测 -> 沉积过程)
@ti.kernel
def Erosion():
    for _ in range(10000):
        indexX = ti.random(int) % sizeX
        indexY = ti.random(int) % sizeY
        # indexX = random.randint(0, sizeX - 1)
        # indexY = random.randint(0, sizeY - 1)
        particlePosition = ti.Vector([indexX * 1.0, indexY * 1.0])
        particleVelocity = ti.Vector([0.0, 0.0])
        particleVolume = 0.1
        particleSediment = 0.0
        while particleVolume > minVolume:
            iPos = particlePosition
            normal = terrainNormal[int(iPos[0]), int(iPos[1])]
            newVelocity = deltaT * ti.Vector([normal[0], normal[1]]) / (particleVolume * density)            
            particleVelocity += newVelocity
            particlePosition += deltaT * newVelocity
            particleVelocity *= (1.0 - deltaT * friction)

            if(particlePosition[0] < 0 or particlePosition[0] > sizeX or 
               particlePosition[1] < 0 or particlePosition[1] > sizeY):
                break

            c_eq = max(0.0, particleVolume * particleVelocity.norm() * (terrainHeight[int(iPos[0]), int(iPos[1])] - terrainHeight[int(particlePosition[0]), int(particlePosition[1])]))
            cDiff = c_eq - particleSediment
            particleSediment += deltaT * KDeposition * cDiff
            oldTerrainHeight = terrainHeight[int(iPos[0]), int(iPos[1])]
            terrainHeight[int(iPos[0]), int(iPos[1])] = oldTerrainHeight - (deltaT * particleVolume * KDeposition * cDiff)
            particleVolume *= (1.0 - deltaT * KEvaporation)



@ti.kernel
def DrawHeight():
    for x, y in terrainHeightMap:
        terrainHeightMap[x, y] = ti.Vector(
            [terrainHeight[x, y]/255.0, terrainHeight[x, y]/255.0, terrainHeight[x, y]/255.0])

if __name__ == "__main__":
    heightFile = open("HeightMap/Height512x512.txt", "r")
    for line in heightFile:
        data = line.split(' ')[:-1]
        heightData.append(list(map(float, data)))
    gui = ti.GUI("Test", res=(sizeX, sizeY))

    GenerateTerrainHeight()
    CalculateTerrainNormal()
    while gui.running:
        Erosion()
        DrawHeight()
        gui.set_image(terrainHeightMap)
        gui.show()


