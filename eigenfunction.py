import numpy.polynomial.polynomial as poly
import numpy as np

def polyKofaktor(M):
    # M adalah sebuah matriks persegi
    # Mengembalikan determinan matriks M (polinom karakteristik matriks M)
    # dengan metode kofaktor rekursif

    # KAMUS LOKAL
    # minorMat : polynomial[][]
    # polyEq, polyTmp : polynomial
    # dim : int
    # i, j, k = int

    # ALGORITMA
    dim = len(M)
    if(dim == 1):
        return M[0][0]
    else:
        minorMat = [[0 for i in range(dim-1)] for i in range(dim-1)]
        polyEq = 0
        for i in range(dim):    
            # Copy matriks minor
            for j in range(1, dim):
                for k in range(i):
                    minorMat[j-1][k] = M[j][k]

            for j in range(1, dim):
                for k in range(i+1, dim):
                    minorMat[j-1][k-1] = M[j][k]

            # hitung polynomial
            polyTmp = poly.polymul(M[0][i], polyKofaktor(minorMat))
            if(i%2 == 0):
                polyTmp = [(-x) for x in polyTmp]
            polyEq = poly.polyadd(polyEq, polyTmp)

        return polyEq
    #


def eigenValue1(A):
    # A adalah sebuah matriks persegi
    # Mengembalikan nilai-nilai eigen melalui akar-akar
    # persamaan karakteristik dari A dengan metode kofaktor

    # KAMUS LOKAL
    # eigMat : polynomial[][]
    # charPol : polynomial
    # i, j : int
    # roots : float[]

    # ALGORITMA
    eigMat = [[El for El in ROW] for ROW in A]  # buat matriks A - λI
    for i in range(0,len(eigMat)):  # Karena eigMat adalah matriks persegi, jumlah baris dan kolom sama
        eigMat[i][i] = (eigMat[i][i], -1)
    
    charPol = polyKofaktor(eigMat)
    roots =  poly.polyroots(charPol)

    roots = list(roots)
    roots.sort(reverse=True)

    i = 0
    while (i < len(roots)):
        for j in range (i+1, len(roots)):
            # maks = max(roots[j], roots[i])
            # minn = min(roots[j], roots[i])
            if (roots[j] > 0):
                elmax = roots[i]
                elmin = roots[j]
            else:
                elmax = roots[j]
                elmin = roots[i]
            if ((elmax-elmin)/elmax <= 1e-6):
                roots[j] = roots[i]
                i = j
        i = i+1

    return roots



def eigenVectors1(A, eigVal):
    # A adalah sebuah matriks persegi
    # mengembalikan vektor-vektor basis eigen A
    # dari nilai eigen eigVal

    # KAMUS LOKAL
    # M, lmdI, ret : float[][]
    # n, npivot : int
    # i, j : int
    # r, c, rpivot : int
    # prec = int


    # ALGORITMA
    dim = len(A)
    lmdI = np.eye(dim) * eigVal
    M = [[El for El in ROW] for ROW in A]
    M = np.array(M)
    M = M - lmdI
    M = np.round_(M, decimals = 16)

    prec = 16
    while True:
        # Eliminasi Gauss-Jordan
        Mtemp = [[El for El in ROW] for ROW in M]
        r=0; c=0
        while(r < dim and c < dim):
            rpivot = r
            while(rpivot < dim-1 and Mtemp[rpivot][c] == 0):
                rpivot = rpivot + 1
            # rpivot = dim-1 or m[rpivot][c] != 0
            if(Mtemp[rpivot][c] != 0):
                # tukar baris dengan pivot
                if(r != rpivot):
                    temp = Mtemp[r]
                    Mtemp[r] = Mtemp[rpivot]
                    Mtemp[rpivot] = temp
                
                # bagi baris pivot dengan elemen bukan 0 pertama
                Mtemp[r] = Mtemp[r]/Mtemp[r][c]

                # eliminasi turun
                for i in range(r+1,dim):
                    Mtemp[i] = Mtemp[i] - Mtemp[r]*Mtemp[i][c]
                
                # eliminasi naik
                for i in range(r):
                    Mtemp[i] = Mtemp[i] - Mtemp[r]*Mtemp[i][c]

                
                Mtemp = np.round_(Mtemp, decimals = prec)
                r = r+1
            c = c+1
        # r == dim or c == dim
        
        if (r != dim):
            break
        else:
            prec = prec - 1
    # iterasi selesai

    r = 0; c = 0
    while (r < dim and c < dim):
        if (Mtemp[r][c] != 0):
            r = r+1
        c = c+1
    # end while
    if (r == 0):
        ret = [0 for i in range(dim)]
        ret[dim-1] = 1
    else:
        n = dim-r   # jumlah vektor basis (null vector) adalah dimensi matriks - jumlah baris bukan 0
        ret = [[0 for i in range(dim)] for j in range(n)]

        nullcol = [0 for i in range(n)] # cari kolom null Mtemp
        npivot = r
        pivotcol = [0 for i in range(npivot)]   # cari kolom pivot Mtemp
        r = 0; c = 0
        while (r < dim and c < dim):
            if (Mtemp[r][c] != 0):
                pivotcol[r] = c
                r = r+1
            else:
                nullcol[c-r] = c
            c = c+1
        # end while

        for i in range(n):
            for j in range(npivot):
                ret[i][pivotcol[j]] = -Mtemp[j][nullcol[i]]
            ret[i][nullcol[i]] = 1

    return ret
    #


def QRsqHouseholder(A):
    # mengembalikan Q, R; faktor A hasil dekomposisi QR
    # QR decomposition menggunakan metode Householder Reflection
    # reference: https://rpubs.com/aaronsc32/qr-decomposition-householder
    #            https://en.wikipedia.org/wiki/QR_decomposition
    #            https://statlect.com/matrix-algebra/Householder-matrix
    #            https://ristohinno.medium.com/qr-decomposition-903e8c61eaab (complete qr decomp function in py, bruh)

    dim = len(A)

    Q = np.eye(dim)
    HA = np.array([[x for x in row] for row in A])
    for i in range(dim):
        u = np.array([x for x in HA[i:,i]]) # u adalah vektor kolom ke-i dari submatriks (Hi x A)
        
        # hitung vektor 
        if(u[0] < 0):   # buat tanda kedua nilai elemen sama untuk menghindari pembagian 0
            u[0] = u[0] - np.linalg.norm(u)
        else:
            u[0] = u[0] + np.linalg.norm(u)

        # normalkan vektor
        normal = np.linalg.norm(u)
        if(normal == 0):
            u[0] = 1
        else:
            u = u/normal
        
        # hitung matriks householder H
        H = np.eye(dim)

        H[i:, i:] = np.eye(dim-i) - 2 * (u[:, None] @ u[None, :])

        Q = Q @ H
        HA = H @ HA

    return Q, HA

def eigen(A, iteration=10000):
    # menghasilkan array eigenvalue matriks A dan matriks vektor-vektor eigen A
    # reference: https://en.wikipedia.org/wiki/QR_algorithm
    #            https://www.andreinc.net/2021/01/25/computing-eigenvalues-and-eigenvectors-using-qr-decomposition

    dim = len(A)
    Ak = np.array([[x for x in row] for row in A])
    QQ = np.eye(dim)

    for i in range(iteration):
        # gunakan metode "shift"
        s = Ak[dim-1,dim-1] * np.eye(dim)

        # dekomposisi matriks A_k yang telah dikurangi dengan "shift"
        (Q, R) = QRsqHouseholder((Ak - s))

        Ak = (R @ Q) + s    # tambahkan "shift" kembali ke matriks A_k+1
        QQ = QQ @ Q

    EigVal = [Ak[i,i] for i in range(dim)]

    # Jika A adalam matriks simetris, QQ adalah matriks kolom vektor eigen dari A
    return EigVal, QQ