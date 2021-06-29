from numba import cuda,int64,float64
import numpy as np


# GPU parameters
TPB = 32

def calcChi(xs, ys, hrad):
	chi = np.zeros(xs.shape)
	chi[(xs>=-hrad) & (ys>=-hrad) & (xs<=1+hrad) & (ys<=0.5+hrad)] = 2
	chi[(xs>=0) & (ys>=0) & (xs<=1) & (ys<=0.5)] = 1
	return chi

def getCartCoords(gfill, hrad):
	L = 1/gfill
	grid_bbox = [[-.2 - L/2, 1.2],[-.2 - L/2, .7]]
	xs = np.arange(grid_bbox[0][0],grid_bbox[0][1], L)
	ys = np.arange(grid_bbox[1][0],grid_bbox[1][1], L)
	XS, YS = np.meshgrid(xs, ys)
	xs, ys = np.reshape(XS,[1, -1]), np.reshape(YS, [1, -1])
	chi = calcChi(xs, ys, hrad)
	rcoords = np.vstack([xs[chi == 1], ys[chi == 1]])
	#fcoords = np.vstack([xs[chi == 2], ys[chi == 2]])
	return rcoords#, fcoords

@cuda.jit
def d_getConn(hrad, d_coords, d_conn):
    i = cuda.grid(1)

    if i<d_coords.shape[1]:
        d_conn[0, i] = i
        cat = 1
        j = 0
        xi, yi = d_coords[:, i]
        while j<d_coords.shape[1] and cat < d_conn.shape[0]:
            if j != i:
                xj, yj = d_coords[:, j]
                dst = ((xi - xj)**2 + (yi - yj)**2)**0.5
                
                if dst<hrad:
                    d_conn[cat, i] = j
                    cat += 1
            j += 1
        for c in range(cat, d_conn.shape[0]):
            d_conn[c, i] = i

@cuda.jit
def d_getInvConn(d_conn, d_iconn):
	# conn[j,i]: index of jth bond of i
	# iconn[j,i]: conn[iconn[j,i], conn[j,i]] = i
	i, j = cuda.grid(2)

	if j<d_conn.shape[0] and i<d_conn.shape[1]:
		jind = d_conn[j, i]
		
		if jind == i:
			d_iconn[j, i] = 0
		else:
			kind = 0
			for k in range(d_conn.shape[0]):
				if d_conn[k, jind] == i:
					kind = k
			d_iconn[j, i] = kind

class Geom:
    def __init__(self, tb, ln, wd, gfill, NB, save = False, load = False):
        if not load:
            # # Generate geometry
            hrad = 3.52/gfill
            coords = getCartCoords(gfill, hrad)
            ND, NN = coords.shape[0],coords.shape[1]
            d_coords = cuda.to_device(coords)
            d_conn = cuda.to_device(np.zeros((NB, coords.shape[1])).astype(np.int32))
            d_getConn[(NN + TPB)//TPB, TPB](hrad, d_coords,d_conn)
            d_iconn = cuda.to_device(np.zeros((NB, NN)).astype(int) - 1)
            d_getInvConn[((NN + TPB)//TPB, (NB + TPB)//TPB),(TPB, TPB)](d_conn, d_iconn)
            dv = tb*ln*wd / coords.shape[1]

            if save:
                conn = d_conn.copy_to_host()
                iconn = d_iconn.copy_to_host()
                np.save('/usr/lusers/jdbart/coords.npy',coords)
                np.save('/usr/lusers/jdbart/conn.npy',conn)
                np.save('/usr/lusers/jdbart/iconn.npy',iconn)
        else:
            coords = np.load('/usr/lusers/jdbart/coords.npy')
            conn = np.load('/usr/lusers/jdbart/conn.npy')
            iconn = np.load('/usr/lusers/jdbart/iconn.npy')
            d_coords = cuda.to_device(coords)
            d_conn =  cuda.to_device(conn)
            d_iconn =  cuda.to_device(iconn)
            dv = tb*ln*wd / coords.shape[1]

        # Apply initial dispacements
        u = np.zeros(coords.shape)
        u[1,(coords[0,:]>(1 - 1/gfill)) & (abs(coords[1,:] - 0.25)<(1/gfill))] = 0.001
        d_u = cuda.to_device(u)
        d_up = cuda.to_device(u)

        self.d_coords = d_coords
        self.d_conn = d_conn
        self.d_iconn = d_iconn
        self.dv = dv
        self.d_u = d_u
        self.d_up = d_up