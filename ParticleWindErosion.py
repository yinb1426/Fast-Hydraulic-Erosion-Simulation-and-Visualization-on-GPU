import random
import time
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
particlesMaxNumber = 10000

pSpeed = ti.Vector([3.0, 3.0, 0.0])

terrainHeight = ti.field(dtype=ti.f32, shape=(sizeX, sizeY))
terrainNormal = ti.Vector.field(3, dtype=ti.f32, shape=(sizeX, sizeY))
sedimentHeight = ti.field(dtype=ti.f32, shape=(sizeX, sizeY))
terrainHeightMap = ti.Vector.field(3, dtype=ti.f32, shape=(sizeX, sizeY))
direction = ti.Vector.field(2, dtype=int, shape=8)
# 初始化地面高度
def GenerateTerrainHeight():
    for x in range(sizeX):
        print(x)
        for y in range(sizeY):
            terrainHeight[x, y] = heightData[x][y]

# 初始化沉积高度
@ti.kernel
def GenerateSedimentHeight():
    for x in range(sizeX):
        for y in range(sizeY):
            sedimentHeight[x, y] = 1.0

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

@ti.func
def Cascade(indexX: int, indexY: int):
    # direction = [ti.Vector[indexX - 1, indexY - 1],
    #              ti.Vector[indexX - 1, indexY],
    #              ti.Vector[indexX - 1, indexY + 1],
    #              ti.Vector[indexX, indexY - 1],
    #              ti.Vector[indexX, indexY + 1],
    #              ti.Vector[indexX + 1, indexY - 1],
    #              ti.Vector[indexX + 1, indexY],
    #              ti.Vector[indexX + 1, indexY + 1],]
    
    direction[0] = ti.Vector([indexX - 1, indexY - 1])
    direction[1] = ti.Vector([indexX - 1, indexY])
    direction[2] = ti.Vector([indexX - 1, indexY + 1])
    direction[3] = ti.Vector([indexX, indexY - 1])
    direction[4] = ti.Vector([indexX, indexY + 1])
    direction[5] = ti.Vector([indexX + 1, indexY - 1])
    direction[6] = ti.Vector([indexX + 1, indexY])
    direction[7] = ti.Vector([indexX + 1, indexY + 1])
    for i in range(8):
        # 边界检测
        if(direction[i][0] < 0 or direction[i][0] > sizeX or 
           direction[i][1] < 0 or direction[i][0] > sizeY):
            continue
        
        # 该位置与临近位置的高程差和与粗糙度的差值
        diff = terrainHeight[indexX, indexY] + sedimentHeight[indexX, indexY] - terrainHeight[direction[i][0], direction[i][1]] - sedimentHeight[direction[i][0], direction[i][1]]
        excess = abs(diff) - sedimentRoughness

        # 稳定状态
        if(excess <= 0): continue

        transfer = 0.0
        
        if(diff > 0):   # 该位置更高
            transfer = min(sedimentHeight[indexX, indexY], excess / 2.0)
        else:   # 临近位置更高
            transfer = min(sedimentHeight[direction[i][0], direction[i][1]], excess / 2.0) * (-1.0)

        deltaHeight = deltaT * settingRate * transfer
        sedimentHeight[indexX, indexY] -= deltaHeight
        sedimentHeight[direction[i][0], direction[i][1]] += deltaHeight

@ti.kernel
def Erosion():
    for _ in range(1000):
        # 风粒子初始属性
        windPosition = ti.Vector([0.0, 0.0])
        windSpeed = pSpeed
        windHeight = 0.0
        windSediment = 0.0

        shift = ti.random(int) % (sizeX + sizeY)
        # shift = random.randint(0, sizeX + sizeY - 1)
        if(shift < sizeX):
            windPosition = ([shift * 1.0, 1.0])
        else:
            windPosition = ([1.0, (shift - sizeX) * 1.0])
        
        # 粒子飞行
        while True:
            iPos = windPosition
            indexX = int(iPos[0])
            indexY = int(iPos[1])
            
            # 如果粒子在地表下面，则设置height为地表
            if(windHeight < terrainHeight[indexX, indexY] + sedimentHeight[indexX, indexY]):
                windHeight = terrainHeight[indexX, indexY] + sedimentHeight[indexX, indexY]
            
            normal = terrainNormal[indexX, indexY]

            # 粒子移动
            
            if(windHeight > terrainHeight[indexX, indexY] + sedimentHeight[indexX, indexY]):    # 粒子高度高于地表，则粒子处于飞行状态，受到重力作用
                windSpeed -= deltaT * ti.Vector([0.0, 0.0, 0.01])
            else:   # 否则，粒子则为在地表滑动碰撞的状态
                acceleration0 = windSpeed.cross(normal)
                windSpeed += deltaT * acceleration0.cross(normal)

            # 粒子受到盛行风作用
            windSpeed += 0.1 * deltaT * (pSpeed - windSpeed)

            # 更新位置
            windPosition += deltaT * ti.Vector([windSpeed[0], windSpeed[1]])
            windHeight += deltaT * windSpeed[2]

            newIndexX = int(windPosition[0])
            newIndexY = int(windPosition[1])

            # 边界检测
            if(windPosition[0] < 0 or windPosition[0] > sizeX or windPosition[1] < 0 or windPosition[1] > sizeY):
                break
            
            # 物质移动(Mass Transport)
            if(windHeight <= terrainHeight[newIndexX, newIndexY] + sedimentHeight[newIndexX, newIndexY]):   # 粒子和地面接触
                force = windSpeed.norm() * (terrainHeight[newIndexX, newIndexY] + sedimentHeight[newIndexX, newIndexY] - windHeight)
                
                if(sedimentHeight[indexX, indexY] <= 0):
                    sedimentHeight[indexX, indexY] = 0.0
                    deltaHeight = deltaT * abrasionRate * force * windSediment
                    terrainHeight[indexX, indexY] -= deltaHeight
                    sedimentHeight[indexX, indexY] += deltaHeight
                elif(sedimentHeight[indexX, indexY] > deltaT * suspensionRate * force):
                    deltaHeight = deltaT * suspensionRate * force
                    sedimentHeight[indexX, indexY] -= deltaHeight
                    windSediment +=deltaHeight
                    Cascade(indexX, indexY)
                else:
                    sedimentHeight[indexX, indexY] = 0
            
            else:   # 粒子飞行
                deltaHeight = deltaT * suspensionRate * windSediment
                windSediment -= deltaHeight
                sedimentHeight[newIndexX, newIndexY] += 0.5 * deltaHeight
                sedimentHeight[indexX, indexY] += 0.5 * deltaHeight
                Cascade(newIndexX, newIndexY)
                Cascade(indexX, indexY)

            # 粒子停下来了
            if(windSpeed.norm() < 0.01):
                break

@ti.kernel
def DrawHeight():
    for x, y in terrainHeightMap:
        terrainHeightMap[x, y] = ti.Vector(
            [(terrainHeight[x, y] + sedimentHeight[x, y])/255.0,
             (terrainHeight[x, y] + sedimentHeight[x, y])/255.0,
             (terrainHeight[x, y] + sedimentHeight[x, y])/255.0])

if __name__ == "__main__":
    heightFile = open("HeightMap/Height512x512.txt", "r")
    for line in heightFile:
        data = line.split(' ')[:-1]
        heightData.append(list(map(float, data)))
    gui = ti.GUI("Test", res=(sizeX, sizeY))

    GenerateTerrainHeight()
    GenerateSedimentHeight()
    CalculateTerrainNormal()

    while gui.running:
        Erosion()
        DrawHeight()
        gui.set_image(terrainHeightMap)
        gui.show()
    
