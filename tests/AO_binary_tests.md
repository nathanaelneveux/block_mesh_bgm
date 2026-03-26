Current Row         1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1

Depth -1 Rows
-1                  1 0 1 0 0 0 0 0 0 1 1 1 1 1 1 1 0 0 0
0                   1 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1
+1                  1 0 1 0 0 1 1 1 1 1 1 1 0 0 0 0 0 0 1

A = CR - (CR ^ 0)   0 1 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 (opaque visibility of current row)

B = A & 0>>         0 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 (unit and v mergable)
C = A & 0<<         0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 (unit and v mergable)
D = B | C           0 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 1 0 (unit and v mergable combine)

E = -1 & 0 & +1     1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
F = A & E>>         0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
G = A & E<<         0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
H = F & G           0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 (final v mergable quads list)

I = +1<< & +1>>     0 1 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 (u merge)
J = -1<< & -1>>     0 1 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 (u merge)
K = +1<< ^ +1>>     0 0 0 1 1 1 0 0 0 0 0 1 1 0 0 0 0 1 0 (unit)
L = -1<< ^ -1>>     0 0 0 1 0 0 0 0 1 1 0 0 0 0 0 1 1 0 0 (unit)

M = (D|K|L)& !H& A  0 0 0 0 1 1 0 0 1 1 0 1 1 0 0 1 1 1 0 (final unit quads list)
N = (I|J)& !H & !M  0 0 0 0 0 0 1 1 0 0 1 0 0 1 1 0 0 0 0
O = N & A           0 0 0 0 0 0 1 1 0 0 1 0 0 1 1 0 0 0 0 (final u mergable quads list)


Match Output        0 N 0 0 U U M M U U M U U M M U U U 0


Second test

        1 0 0 0 0 0 0
        1 0 0 1 0 0 1
        1 0 0 0 0 0 1

A =     0 1 1 0 1 1 0
B =     0 1 0 0 1 0 0
C =     0 0 1 0 0 1 0

G =     0 1 0 0 0 1 0