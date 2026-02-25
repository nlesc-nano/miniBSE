import numpy as np

def davidson(matvec, diag, nroots, max_iter=500, tol=1e-6, max_subspace=None):
    
    n = len(diag)
    # Dynamically set max subspace if not provided (5x roots is standard)
    if max_subspace is None:
        max_subspace = max(50, 5 * nroots)

    # Initialize subspace with random vectors
    V = np.random.rand(n, nroots)
    V, _ = np.linalg.qr(V)

    # Pre-compute AV to save matvec calls in the loop
    AV = np.column_stack([matvec(V[:, i]) for i in range(V.shape[1])])

    for it in range(max_iter):

        # Subspace Hamiltonian
        Hsub = V.T @ AV

        # Diagonalize the small subspace Hamiltonian
        evals, evecs = np.linalg.eigh(Hsub)
        
        # Get Ritz vectors and their matvecs for the lowest nroots
        Ritz = V @ evecs[:, :nroots]
        Ritz_AV = AV @ evecs[:, :nroots]

        # Compute residuals
        residuals = np.zeros((n, nroots))
        for i in range(nroots):
            residuals[:, i] = Ritz_AV[:, i] - evals[i] * Ritz[:, i]

        norms = np.linalg.norm(residuals, axis=0)
        print(f"[DAV] Iter {it:3d} residuals:", np.round(norms, 6))

        # Check convergence
        if np.all(norms < tol):
            print(f"[DAV] Converged in {it} iterations.")
            return evals[:nroots], Ritz

        # Check if subspace needs to collapse
        if V.shape[1] >= max_subspace:
            # Collapse back down to just the best current Ritz vectors
            V = Ritz
            V, _ = np.linalg.qr(V)
            # Must recompute AV after a collapse
            AV = np.column_stack([matvec(V[:, i]) for i in range(V.shape[1])])
            continue

        # Generate new correction vectors
        new_vecs = []
        for i in range(nroots):
            if norms[i] > tol:
                # Preconditioner
                diff = diag - evals[i]
                
                # Safeguard against denominator singularity!
                diff[np.abs(diff) < 1e-4] = 1e-4
                
                delta = residuals[:, i] / diff
                new_vecs.append(delta)

        # Modified Gram-Schmidt orthogonalization
        for delta in new_vecs:
            # Orthogonalize against all existing vectors in V
            for j in range(V.shape[1]):
                overlap = np.dot(V[:, j], delta)
                delta -= overlap * V[:, j]
            
            norm = np.linalg.norm(delta)
            # Only add to subspace if it is sufficiently linearly independent
            if norm > 1e-5:
                delta /= norm
                V = np.column_stack((V, delta))
                # Compute matvec ONLY for the new vector and append to AV
                AV = np.column_stack((AV, matvec(delta)))

    raise RuntimeError(f"Davidson did not converge after {max_iter} iterations. Max residual: {np.max(norms):.2e}")

