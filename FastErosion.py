import taichi as ti
import taichi.math as tm

ti.init(arch=ti.gpu)

# 这一套参数，效果OK
sizeX = 1024     # 格网尺寸X
sizeY = 1024     # 格网尺寸Y
deltaT = 0.05    # 时间间隔(很有影响)
pipeLength = 1  # 管道长度
gravity = 9.8
rainfallRate = 0.1      # 降雨速率
rainfallBoundary = [280, 780, 280, 780]     # 降雨速率遮罩边界：左，右，下，上
waterTopHeight = 150.0      # 水面顶部高度
heightData = []
Kc = 0.01
Ks = 0.04
Kd = 0.04
Ke = 0.001
minTiltSlope = tm.pi / 16.0 # 45/4° 最小倾斜坡度
# cellX = cellY = 1

terrainHeight = ti.field(dtype=ti.f32, shape=(sizeX, sizeY))     # 地面高度b
waterHeight = ti.field(dtype=ti.f32, shape=(sizeX, sizeY))       # 水面高度d
totalHeight = ti.field(dtype=ti.f32, shape=(sizeX, sizeY))        # 总高度b+d
rainfallMask = ti.field(dtype=ti.f32, shape=(sizeX, sizeY))      # 降雨速率r(x,y)
terrainHeightMap = ti.Vector.field(3, dtype=ti.f32, shape=(sizeX, sizeY))
outputFlow = ti.Vector.field(4, dtype=ti.f32, shape=(sizeX, sizeY))   # 流量场(L,R,T,B)
newOutputFlow = ti.Vector.field(4, dtype=ti.f32, shape=(sizeX, sizeY))   # 临时存放的新流量场(L,R,T,B)
waterVelocity = ti.Vector.field(2, dtype=ti.f32, shape=(sizeX,sizeY))    # 水流速度
sedimentHeight = ti.field(dtype=ti.f32, shape=(sizeX, sizeY))   # 沉积物高度
tempSediment = ti.field(dtype=ti.f32, shape=(sizeX, sizeY))     # 用于MacCormack方法使用的临时沉积高度

# 初始化地面高度
def GenerateTerrainHeight():
    for x in range(sizeX):
        print(x)
        for y in range(sizeY):
            terrainHeight[x, y] = heightData[x][y]

# 初始化降雨速率遮罩（可使用高斯函数生成降雨遮罩矩阵）
@ti.kernel
def GenerateRainfall():
    for x, y in rainfallMask:
        if(x > rainfallBoundary[0] and x < rainfallBoundary[1] and y > rainfallBoundary[2] and y < rainfallBoundary[3]):
            rainfallMask[x, y] = rainfallRate
        else:
            rainfallMask[x, y] = 0.0

# 初始化水面高度
@ti.kernel
def GenerateWaterHeight():
    for x, y in waterHeight:
        if(terrainHeight[x, y] < waterTopHeight):
            waterHeight[x, y] = waterTopHeight - terrainHeight[x, y]
        else:
            waterHeight[x, y] = 0.0

# 初始化沉积物高度
@ti.kernel
def GenerateSedimentHeight():
    for x, y in sedimentHeight:
        sedimentHeight[x, y] = 0.0
        tempSediment[x, y] = 0.0

# 初始化流量场
@ti.kernel
def InitOutputFlow():
    for x, y in outputFlow:
        outputFlow[x, y] = ti.Vector([0.0, 0.0, 0.0, 0.0])
        newOutputFlow[x, y] = ti.Vector([0.0, 0.0, 0.0, 0.0])

# 初始化水流速度场
@ti.kernel
def InitVelocity():
    for x, y in outputFlow:
        waterVelocity[x, y] = ti.Vector([0.0, 0.0])

# Step1: 降雨导致水面升高 Section(3.1)
@ti.kernel
def WaterIncrement():
    for x, y in waterHeight:
        waterHeight[x, y] = waterHeight[x, y] + deltaT * rainfallMask[x, y]     # eq.(1)

# Step2: 水流模拟 Section(3.2.1)
@ti.kernel
def UpdateOutputFlow():
    for x in range(1, sizeX - 1):
        for y in range(1, sizeY - 1):
            # eq.(3)
            deltaHeightLeft = terrainHeight[x, y] + waterHeight[x, y] - terrainHeight[x-1, y] - waterHeight[x-1, y]
            deltaHeightRight = terrainHeight[x, y] + waterHeight[x, y] - terrainHeight[x+1, y] - waterHeight[x+1, y]
            deltaHeightTop = terrainHeight[x, y] + waterHeight[x, y] - terrainHeight[x, y-1] - waterHeight[x, y-1]
            deltaHeightBottom = terrainHeight[x, y] + waterHeight[x, y] - terrainHeight[x, y+1] - waterHeight[x, y+1]

            oldOutputFlowLeft = outputFlow[x, y][0]
            oldOutputFlowRight = outputFlow[x, y][1]
            oldOutputFlowTop = outputFlow[x, y][2]
            oldOutputFlowBottom = outputFlow[x, y][3]

            damping = 0.999
            
            # eq.(2)
            newOutputFlowLeft = max(0, damping * oldOutputFlowLeft + deltaT * pipeLength * pipeLength * gravity * deltaHeightLeft / pipeLength)
            newOutputFlowRight = max(0, damping * oldOutputFlowRight + deltaT * pipeLength * pipeLength * gravity * deltaHeightRight / pipeLength)
            newOutputFlowTop = max(0, damping * oldOutputFlowTop + deltaT * pipeLength * pipeLength * gravity * deltaHeightTop / pipeLength)
            newOutputFlowBottom = max(0, damping * oldOutputFlowBottom + deltaT * pipeLength * pipeLength * gravity * deltaHeightBottom / pipeLength)

            outputVolume = (newOutputFlowLeft + newOutputFlowRight + newOutputFlowTop + newOutputFlowBottom) * deltaT
            K = min(1, waterHeight[x, y] * pipeLength * pipeLength / outputVolume)      # eq.(4)

            # eq.(5)
            newOutputFlowLeft *= K
            newOutputFlowRight *= K
            newOutputFlowTop *= K
            newOutputFlowBottom *= K

            newOutputFlow[x, y] = ti.Vector([newOutputFlowLeft, newOutputFlowRight, newOutputFlowTop, newOutputFlowBottom])
    for x,y in outputFlow:
        outputFlow[x,y] = newOutputFlow[x,y]
        newOutputFlow[x,y] = ti.Vector([0.0, 0.0, 0.0, 0.0])

# Step3: 更新水体的速度和高度 Section(3.2.2)
@ti.kernel
def UpdateVelocityAndWaterHeight():
    for x in range(1, sizeX - 1):
        for y in range(1, sizeY - 1):
            deltaV = (outputFlow[x-1,y][1] + outputFlow[x+1,y][0] + outputFlow[x,y-1][3] + outputFlow[x,y+1][2] - outputFlow[x,y][0] - outputFlow[x,y][1] - outputFlow[x,y][2] - outputFlow[x,y][3]) * deltaT       # eq.(6)
            d2 = waterHeight[x, y] + deltaV / (pipeLength * pipeLength)     # eq.(7)
            averageWaterHeight = (d2 + waterHeight[x,y]) / 2.0
            velocityFactor = averageWaterHeight * pipeLength

            # eq.(8)
            deltaWX = (outputFlow[x-1,y][1] - outputFlow[x,y][0] + outputFlow[x,y][1] - outputFlow[x+1,y][0]) / 2.0
            deltaWY = (outputFlow[x,y-1][3] + outputFlow[x,y][3] - outputFlow[x,y][2] - outputFlow[x,y+1][2]) / 2.0
            
            # eq.(9)
            velocityU = deltaWX / velocityFactor if velocityFactor > 0.0005 else 0.0
            velocityV = deltaWY / velocityFactor if velocityFactor > 0.0005 else 0.0
            
            waterHeight[x,y] = d2
            waterVelocity[x,y] = ti.Vector([velocityU,velocityV])

# Step4: 侵蚀和搬运 Section(3.3)
@ti.kernel
def ErosionAndDeposition():
    for x in range(1, sizeX - 1):
        for y in range(1, sizeY - 1):
            deltaHeightX = terrainHeight[x+1, y] - terrainHeight[x-1, y]
            deltaHeightY = terrainHeight[x, y+1] - terrainHeight[x, y-1]
            R2L = ti.Vector([2.0, 0.0, deltaHeightX])
            B2T = ti.Vector([0.0, 2.0, deltaHeightY])
            normal0 = R2L.cross(B2T)
            normal = normal0 / normal0.norm()
            slope = abs(tm.acos(normal[2] / tm.sqrt(normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2])))
            slope = max(slope, minTiltSlope)
            
            # eq.(10)
            sedimentCapacity = Kc * tm.sin(slope) * tm.sqrt(waterVelocity[x,y][0] * waterVelocity[x,y][0] + waterVelocity[x,y][1] * waterVelocity[x,y][1])
            
            # eq.(11)-(12)
            newTerrainHeight = 0.0
            newSedimentHeight = 0.0
            if(sedimentCapacity > sedimentHeight[x,y]):
                newTerrainHeight = terrainHeight[x,y] - Ks * (sedimentCapacity - sedimentHeight[x,y])
                newSedimentHeight = sedimentHeight[x,y] + Ks * (sedimentCapacity - sedimentHeight[x,y])
            else:
                newTerrainHeight = terrainHeight[x,y] + Kd * (sedimentHeight[x,y] - sedimentCapacity)
                newSedimentHeight = sedimentHeight[x,y] - Kd * (sedimentHeight[x,y] - sedimentCapacity)
            terrainHeight[x,y] = newTerrainHeight
            sedimentHeight[x,y] = newSedimentHeight

@ti.func
def frac(n: ti.f32) -> ti.f32:
    return n - tm.floor(n)

@ti.func 
def lerp(n1: ti.f32, n2: ti.f32, t: ti.f32) -> ti.f32:
    return n1 * (1.0 - t) + n2 * t

# Step5: 沉积物搬运(使用MacCormack方法计算) Section(3.4)
@ti.kernel
def SedimentTransportation():
    LTSediment = 0.0 
    RTSediment = 0.0
    LBSediment = 0.0
    RBSediment = 0.0
    for x in range(1, sizeX - 1):
        for y in range(1, sizeY - 1):
            previousX = x - waterVelocity[x,y][0] * deltaT
            previousY = y - waterVelocity[x,y][1] * deltaT
            LTSediment = sedimentHeight[int(tm.floor(previousX)), int(tm.floor(previousY))]
            RTSediment = sedimentHeight[int(tm.ceil(previousX)), int(tm.floor(previousY))]
            LBSediment = sedimentHeight[int(tm.floor(previousX)), int(tm.ceil(previousY))]
            RBSediment = sedimentHeight[int(tm.ceil(previousX)), int(tm.ceil(previousY))]
            BackwardSediment = lerp(lerp(LTSediment, RTSediment, frac(previousX)), lerp(LBSediment, RBSediment, frac(previousX)), frac(previousY))
            tempSediment[x, y] = BackwardSediment
    for x in range(1, sizeX - 1):
        for y in range(1, sizeY - 1):
            nextX = x + waterVelocity[x,y][0] * deltaT
            nextY = y + waterVelocity[x,y][1] * deltaT
            LTSediment = tempSediment[int(tm.floor(nextX)), int(tm.floor(nextY))]
            RTSediment = tempSediment[int(tm.ceil(nextX)), int(tm.floor(nextY))]
            LBSediment = tempSediment[int(tm.floor(nextX)), int(tm.ceil(nextY))]
            RBSediment = tempSediment[int(tm.ceil(nextX)), int(tm.ceil(nextY))]
            BackForwardSediment = lerp(lerp(LTSediment, RTSediment, frac(nextX)), lerp(LBSediment, RBSediment, frac(nextX)), frac(nextY))
            newSediment = tempSediment[x, y] + (sedimentHeight[x, y] - BackForwardSediment) * 0.5
            clampMin = min(min(min(LTSediment, RTSediment), LBSediment), RBSediment)
            clampMax = max(max(max(LTSediment, RTSediment), LBSediment), RBSediment)
            newSediment = max(min(newSediment, clampMax), clampMin)
            sedimentHeight[x, y] = newSediment

# Step6: 蒸发过程 Section(3.5)
@ti.kernel
def Evaporation():
    for x in range(1, sizeX - 1):
        for y in range(1, sizeY - 1):
            oldWaterHeight = waterHeight[x, y]
            newWaterHeight = oldWaterHeight * (1 - Ke * deltaT)     # eq.(15)
            waterHeight[x, y] = 0 if newWaterHeight < 0 else newWaterHeight

@ti.kernel
def DrawHeight():
    for x, y in terrainHeightMap:
        terrainHeightMap[x, y] = ti.Vector(
            [terrainHeight[x, y]/255.0, terrainHeight[x, y]/255.0, terrainHeight[x, y]/255.0])

@ti.kernel
def DrawRainHeight():
    for x, y in terrainHeightMap:
        terrainHeightMap[x, y] = ti.Vector(
            [waterHeight[x, y]/200.0, waterHeight[x, y]/200.0, waterHeight[x, y]/200.0])

@ti.kernel
def DrawSedimentHeight():
    for x, y in terrainHeightMap:
        terrainHeightMap[x, y] = ti.Vector(
            [sedimentHeight[x, y]/50.0, sedimentHeight[x, y]/50.0, sedimentHeight[x, y]/50.0])

if __name__ == "__main__":
    heightFile = open("HeightMap/Height1024x1024.txt", "r")
    for line in heightFile:
        data = line.split(' ')[:-1]
        heightData.append(list(map(float, data)))
    gui = ti.GUI("Test", res=(sizeX, sizeY))

    GenerateTerrainHeight()
    GenerateWaterHeight()
    GenerateSedimentHeight()
    GenerateRainfall()
    InitOutputFlow()
    DrawHeight()
    while gui.running:
        WaterIncrement()
        UpdateOutputFlow()
        UpdateVelocityAndWaterHeight()
        ErosionAndDeposition()
        SedimentTransportation()
        Evaporation()
        DrawHeight()
        # DrawRainHeight()
        # gui.set_image(outputFlow)
        gui.set_image(terrainHeightMap)
        gui.show()
