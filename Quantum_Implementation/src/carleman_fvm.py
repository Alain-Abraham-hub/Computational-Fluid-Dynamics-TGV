import numpy as np
import scipy.sparse as sp

class CarlemanFVM:
    def __init__(self, nx, ny, L, nu):
        """
        Finite Volume Method setup for 2D Taylor-Green Vortex mapped to Carleman Linearization.
        """
        self.nx = nx
        self.ny = ny
        self.dx = L / nx
        self.dy = L / ny
        self.nu = nu
        self.dof = 2 * nx * ny 
        
    def _build_diffusion_operator_A(self):
        """Builds the linear viscous diffusion matrix A."""
        diagonals = [[1.0]*self.nx, [-2.0]*self.nx, [1.0]*self.nx]
        D1x = sp.diags(diagonals, [-1, 0, 1], shape=(self.nx, self.nx), dtype=float).tolil()
        D1x[0, -1] = 1.0; D1x[-1, 0] = 1.0
        
        D1y = sp.diags(diagonals, [-1, 0, 1], shape=(self.ny, self.ny), dtype=float).tolil()
        D1y[0, -1] = 1.0; D1y[-1, 0] = 1.0
        
        Ix = sp.eye(self.nx); Iy = sp.eye(self.ny)
        Lap2D = sp.kron(D1x, Iy) / (self.dx**2) + sp.kron(Ix, D1y) / (self.dy**2)
        Lap2D *= self.nu
        
        A = sp.block_diag((Lap2D, Lap2D))
        return A

    def _build_advection_operator_H(self):
        """Builds the non-linear convective flux tensor H."""
        N = self.nx * self.ny
        dof = self.dof
        H = sp.lil_matrix((dof, dof**2))
        
        def idx(i, j):
            return (j % self.ny) * self.nx + (i % self.nx)
            
        for j in range(self.ny):
            for i in range(self.nx):
                k = idx(i, j)
                k_right = idx(i+1, j); k_left = idx(i-1, j)
                k_up = idx(i, j+1); k_down = idx(i, j-1)
                
                # --- U Equation ---
                u2_right_idx = k_right * dof + k_right
                u2_left_idx = k_left * dof + k_left
                H[k, u2_right_idx] -= 1.0 / (2 * self.dx)
                H[k, u2_left_idx] += 1.0 / (2 * self.dx)
                
                uv_up_idx = k_up * dof + (N + k_up)
                uv_down_idx = k_down * dof + (N + k_down)
                H[k, uv_up_idx] -= 1.0 / (2 * self.dy)
                H[k, uv_down_idx] += 1.0 / (2 * self.dy)
                
                # --- V Equation ---
                uv_right_idx = k_right * dof + (N + k_right)
                uv_left_idx = k_left * dof + (N + k_left)
                H[N + k, uv_right_idx] -= 1.0 / (2 * self.dx)
                H[N + k, uv_left_idx] += 1.0 / (2 * self.dx)
                
                v2_up_idx = (N + k_up) * dof + (N + k_up)
                v2_down_idx = (N + k_down) * dof + (N + k_down)
                H[N + k, v2_up_idx] -= 1.0 / (2 * self.dy)
                H[N + k, v2_down_idx] += 1.0 / (2 * self.dy)
                
        return H.tocsr()

    def build_carleman_system(self, dt):
        """Builds the Order-2 Carleman matrix Ac and discretizes in time."""
        A = self._build_diffusion_operator_A()
        H = self._build_advection_operator_H()
        A_2 = sp.kronsum(A, A)
        
        Ac_top = sp.hstack([A, H])
        Ac_bottom = sp.hstack([sp.csr_matrix((self.dof**2, self.dof)), A_2])
        Ac = sp.vstack([Ac_top, Ac_bottom])
        
        I_Ac = sp.eye(Ac.shape[0])
        M = I_Ac - dt * Ac
        return M

    def build_pressure_laplacian(self):
        """Builds the classical Laplacian matrix for the Pressure Poisson Equation."""
        diagonals = [[1.0]*self.nx, [-2.0]*self.nx, [1.0]*self.nx]
        D1x = sp.diags(diagonals, [-1, 0, 1], shape=(self.nx, self.nx), dtype=float).tolil()
        D1x[0, -1] = 1.0; D1x[-1, 0] = 1.0
        D1y = sp.diags(diagonals, [-1, 0, 1], shape=(self.ny, self.ny), dtype=float).tolil()
        D1y[0, -1] = 1.0; D1y[-1, 0] = 1.0
        
        Ix = sp.eye(self.nx); Iy = sp.eye(self.ny)
        Lap2D = sp.kron(D1x, Iy) / (self.dx**2) + sp.kron(Ix, D1y) / (self.dy**2)
        
        # Pin one node to make the matrix non-singular
        Lap2D = Lap2D.tolil()
        Lap2D[0, :] = 0
        Lap2D[0, 0] = 1.0
        return Lap2D.tocsr()

    def compute_divergence(self, u, v):
        """Computes div(u) using central differences."""
        u = u.reshape((self.ny, self.nx))
        v = v.reshape((self.ny, self.nx))
        div = np.zeros_like(u)
        
        for j in range(self.ny):
            for i in range(self.nx):
                u_right = u[j, (i+1)%self.nx]
                u_left  = u[j, (i-1)%self.nx]
                v_up    = v[(j+1)%self.ny, i]
                v_down  = v[(j-1)%self.ny, i]
                div[j, i] = (u_right - u_left)/(2*self.dx) + (v_up - v_down)/(2*self.dy)
        return div.flatten()
        
    def compute_gradient(self, p):
        """Computes grad(p) using central differences."""
        p = p.reshape((self.ny, self.nx))
        grad_u = np.zeros_like(p)
        grad_v = np.zeros_like(p)
        for j in range(self.ny):
            for i in range(self.nx):
                p_right = p[j, (i+1)%self.nx]
                p_left  = p[j, (i-1)%self.nx]
                p_up    = p[(j+1)%self.ny, i]
                p_down  = p[(j-1)%self.ny, i]
                grad_u[j, i] = (p_right - p_left)/(2*self.dx)
                grad_v[j, i] = (p_up - p_down)/(2*self.dy)
        return grad_u.flatten(), grad_v.flatten()

    def encode_initial_state(self, u_0, v_0):
        """Encodes classical fields into the Carleman state vector y."""
        x = np.concatenate([u_0.flatten(), v_0.flatten()])
        y_0 = np.concatenate([x, np.kron(x, x)])
        return y_0

    def decode_state(self, y):
        """Extracts the classical velocity vector from the quantum/Carleman state."""
        N = self.nx * self.ny
        u = y[0:N]
        v = y[N:2*N]
        return u, v
