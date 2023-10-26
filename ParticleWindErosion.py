import random
import taichi as ti

ti.init(arch=ti.gpu)

sizeX = 512
sizeY = 512

heightData = []

deltaT = 0.25
suspensionRate = 0.0001
abrasionRate = 0.0001
sedimentRoughness = 0.005
settingRate = 0.01

pSpeed = ti.Vector([1.0, 1.0, 0.0])

terrainHeight = ti.field(dtype=ti.f32, shape=(sizeX, sizeY))
terrainNormal = ti.Vector.field(3, dtype=ti.f32, shape=(sizeX, sizeY))
sedimentHeight = ti.field(dtype=ti.f32, shape=(sizeX, sizeY))

# 初始化地面高度
def GenerateTerrainHeight():
    for x in range(sizeX):
        print(x)
        for y in range(sizeY):
            terrainHeight[x, y] = heightData[x][y]

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

def Erosion():
    # 风粒子初始属性
    windPosition = ti.Vector([0.0, 0.0])
    windNewPosition = windPosition
    windSpeed = pSpeed
    windHeight = 0.0
    windSediment = 0.0

    shift = random.randint(0, sizeX + sizeY - 1)
    if(shift < sizeX):
        windPosition = ([shift * 1.0, 1.0])
        windNewPosition = windPosition
    else:
        windPosition = ([1.0, (shift - sizeX) * 1.0])
        windNewPosition = windPosition
    print(1)

if __name__ == "__main__":
    Erosion()
    
