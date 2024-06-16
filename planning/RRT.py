import math
import random
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(projection = '3d')

class env:
    def __init__(self,ob,end,goal_epsilon,RangeBounded, obvertex):
        self.ob = ob
        self.obvertex = obvertex
        self.end = end
        self.epsilon = goal_epsilon
        self.xgmin = end[0] - goal_epsilon
        self.xgmax = end[0] + goal_epsilon
        self.ygmin = end[1] - goal_epsilon
        self.ygmax = end[1] + goal_epsilon
        self.zgmin = end[2] - goal_epsilon
        self.zgmax = end[2] + goal_epsilon
        self.RangeBound = RangeBounded
    
    def cubedraw(self,obsx,obsy,obzl,obzh,k):
        x = obsx
        y = obsy
        zl = [obzl,obzl,obzl,obzl,obzl]
        zh = [obzh,obzh,obzh,obzh,obzh]
		
        ax.plot(x, y, zl, k) #for x y
        ax.plot(x, y, zh, k)
        for i in range (len(x)-1): #for z
            obx = [x[i],x[i]]
            oby = [y[i],y[i]]
            obz = [zl[i],zh[i]]
            ax.plot(obx,oby,obz,k)

class rrt(env):
    def __init__(self, start, dmax, ob, end, goal_epsilon, RangeBounded, BoxToHand, CollisionScale, obvertex):
        super().__init__(ob, end, goal_epsilon, RangeBounded, obvertex)
        (x, y, z) = start
        self.dmax = dmax
        self.x = []
        self.y = []
        self.z = []
        self.parent = []
        self.x.append(x)
        self.y.append(y)
        self.z.append(z)
        self.parent.append(0)
        self.path = []
        self.smooth_path = []
        self.BoxToHand = BoxToHand
        self.CollisionScale = CollisionScale    
    
    # topmost function
    def FindPath(self, nmax = 10000, slice_path_check = False, step = 10, Draw = False):       
        smooth_path = []
        return_path = []
        
        for i in range(0,nmax):
            if i%10 != 0: self.expand()
            else: self.bias()

            if self.inGoal():
                print ("found")
                break
        
        self.path_to_goal()
        self.smooth()
        for i in self.smooth_path[::-1]:
            smooth_path.append([self.x[i], self.y[i], self.z[i]])
        
        if slice_path_check:        
            slice_path = []
            
            for i in range(len(smooth_path) - 1):   
                for j in range(step + 1):
                    per = j/step # %
                    x = smooth_path[i][0]*(1-per) + smooth_path[i+1][0]*per
                    y = smooth_path[i][1]*(1-per) + smooth_path[i+1][1]*per
                    z = smooth_path[i][2]*(1-per) + smooth_path[i+1][2]*per
                    
                    slice_path.append([x, y, z])
                    
            return_path = slice_path
        
        else:
            return_path = smooth_path

        # draw
        if Draw: self.draw() # To visualize the path
        
        return  return_path

    # set up function
    def get_node_num(self):
        return len(self.x)
    
    def remove_node(self, n):
        self.x.pop(n)
        self.y.pop(n)
        self.z.pop(n)

    def add_node(self, n, x, y, z):
        self.x.insert(n, x)
        self.y.insert(n, y)
        self.z.insert(n, z)
    
    def remove_edge(self, n):
        self.parent.pop(n)
    
    def add_edge(self, parent, child):
        self.parent.insert(child, parent)
        
    def Collision(self, x, y, z):
        tmp = [x, y, z]
        for ob in self.obvertex:
            for vertex in ob:
                if((((tmp[0] - vertex[0])**2 + (tmp[1] - vertex[1])**2 + (tmp[2] - vertex[2])**2)**(0.5)) <= self.CollisionScale):
                    return True
        
        return False

    def inObstacle(self, x1, y1, x2, y2, z1, z2): #check connect
        for j in range(101):
            per = j/50.0 # %
            x = x1*per + x2*(1-per)
            y = y1*per + y2*(1-per)
            z = z1*per + z2*(1-per) + self.BoxToHand
            if(self.Collision(x, y, z)):
                return True
            
        return False
    
    def isFree(self, n):
        (x, y, z) = (self.x[n], self.y[n], self.z[n])
        for i in self.ob:
            xomin = i[0]
            xomax = i[1]
            yomin = i[2]
            yomax = i[3]
            zomin = i[4]
            zomax = i[5]
            if(x >= xomin) and (x <= xomax) and (y >= yomin) and (y <= yomax) and (z >= zomin) and (z <= zomax):
                self.remove_node(n)
                return False
            
        return True
        
    def inGoal(self):
        n = self.get_node_num() - 1
        (x, y, z) = (self.x[n], self.y[n], self.z[n])
        if(x >= self.xgmin) and (x <= self.xgmax) and (y >= self.ygmin) and (y <= self.ygmax) and (z >= self.zgmin) and (z <= self.zgmax):
            return True
        else: return False
    
    def distance(self, n1, n2):
        (x1, y1, z1) = (float(self.x[n1]),float(self.y[n1]),float(self.z[n1]))
        (x2, y2, z2) = (float(self.x[n2]),float(self.y[n2]),float(self.z[n2]))

        return ((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)**(0.5)
    
    def near(self, n):
        current_dmin = self.distance(0, n)
        nnear = 0
        for i in range(n):
            cur_dis = self.distance(i, n)
            if(cur_dis < current_dmin):
                current_dmin = cur_dis
                nnear = i
            
        return nnear
    
    def step(self, nnear, nrand):
        dis = self.distance(nnear, nrand)

        if (dis > self.dmax):
            (xnear,ynear,znear)= (self.x[nnear],self.y[nnear],self.z[nnear])
            (xrand,yrand,zrand)= (self.x[nrand],self.y[nrand],self.z[nrand])
            (px,py,pz)=(xrand-xnear,yrand-ynear,zrand-znear)
            theta = math.atan2(py,px)
            x=xnear+self.dmax*math.cos(theta)
            y=ynear+self.dmax*math.sin(theta)
            alpha = math.atan2(pz,y)
            z=znear+self.dmax*math.sin(alpha)
            self.remove_node(nrand)
            self.add_node(nrand,x,y,z)
    
    def connent(self, n1, n2):
        (x1,y1,z1)= (self.x[n1],self.y[n1],self.z[n1])
        (x2,y2,z2)= (self.x[n2],self.y[n2],self.z[n2])
        if self.inObstacle(x1,y1,x2,y2,z1,z2):
            self.remove_node(n2)
        else:
            self.add_edge(n1, n2)

    # find the path
    def expand(self):
        x = random.uniform(self.RangeBound[0], self.RangeBound[1])
        y = random.uniform(self.RangeBound[2], self.RangeBound[3])
        z = random.uniform(self.RangeBound[4], self.RangeBound[5])
        n = self.get_node_num()
        self.add_node(n, x, y, z)

        if(self.isFree(n)):
            nnear = self.near(n)
            self.step(nnear, n)
            self.connent(nnear, n)
    
    def bias(self): # find goal faster
        n = self.get_node_num()
        self.add_node(n, self.end[0], self.end[1], self.end[2])

        nnear = self.near(n)
        self.step(nnear, n)
        self.connent(nnear, n)

    # when find goal
    def path_to_goal(self):
        goalstate = 0
        for i in range(self.get_node_num()):
            (x, y, z) = (self.x[i],self.y[i],self.z[i])
            if (x>=self.xgmin) and (x<=self.xgmax) and (y>=self.ygmin) and (y<=self.ygmax) and (z>=self.zgmin) and (z<=self.zgmax):
                goalstate = i
                break
            
        self.path.append(goalstate)
        pparent = self.parent[goalstate]
        while(pparent != 0):
            self.path.append(pparent)
            pparent = self.parent[pparent]
        
        self.path.append(0) # add start point
    
    def smooth(self): # return smooth path node "index" 
        s = 0 #s for start point
        self.smooth_path.append(self.path[s])
        for i in range(1, len(self.path)-1):
            (x1,y1,z1)=(self.x[self.path[s]],self.y[self.path[s]],self.z[self.path[s]])
            (x2,y2,z2)=(self.x[self.path[i]],self.y[self.path[i]],self.z[self.path[i]])
            if  self.inObstacle(x1, y1, x2, y2, z1, z2):
                self.smooth_path.append(self.path[i - 1])
                s = i - 1
        
        self.smooth_path.append(self.path[-1])
        
    # draw
    def showtree(self,k):
        for i in range (self.get_node_num()):
            par=self.parent[i]
            x=[self.x[i],self.x[par]]
            y=[self.y[i],self.y[par]]
            z=[self.z[i],self.z[par]]
            ax.plot(x,y,z,k,lw=0.5)
    
    def showpath(self,k):
        for i in range (len(self.path)-1):
            n1=self.path[i]
            n2=self.path[i+1]
            x=[self.x[n1],self.x[n2]]
            y=[self.y[n1],self.y[n2]]
            z=[self.z[n1],self.z[n2]]
            ax.plot(x,y,z,k,lw=1,markersize=3)
    
    def showsmooth(self,k):
        for i in range (len(self.smooth_path)-1):
            n1=self.smooth_path[i]
            n2=self.smooth_path[i+1]
            x=[self.x[n1],self.x[n2]]
            y=[self.y[n1],self.y[n2]]
            z=[self.z[n1],self.z[n2]]
            ax.plot(x,y,z,k,lw=2,markersize=5)
    
    def draw(self):
        # gx=[self.xgmin,self.xgmax,self.xgmax,self.xgmin,self.xgmin]
        # gy=[self.ygmin,self.ygmax,self.ygmin,self.ygmax,self.ygmin]
        # self.cubedraw(gx,gy,self.xgmin,self.xgmax,'g')
        
        self.showtree('0.45')
        self.showpath('ro-')
        self.showsmooth('g*-')
        
        for i in self.ob:
            obx=[i[0],i[0],i[1],i[1],i[0]]
            oby=[i[2],i[3],i[3],i[2],i[2]]
            obzmin = i[4]
            obzmax = i[5]
            self.cubedraw(obx,oby,obzmin,obzmax,'k')
        
        plt.show()