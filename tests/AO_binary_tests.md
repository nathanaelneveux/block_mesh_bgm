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
H=(F| G)& !-1& !+1  0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 (final v mergable quads list)

J = -1<< & -1>>     0 1 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 (u merge)
I = +1<< & +1>>     0 1 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 (u merge)
L = -1<< ^ -1>>     0 0 0 1 1 1 0 0 0 0 0 1 1 0 0 0 0 1 0 (unit)
K = +1<< ^ +1>>     0 0 0 1 0 0 0 0 1 1 0 0 0 0 0 1 1 0 0 (unit)
P = -1 & !J         1 0 1 0 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 (possible unit)
Q = +1 & !I         1 0 1 0 0 1 0 0 0 0 0 1 0 0 0 0 0 0 1 (possible unit)
R = K | L           0 0 0 1 1 1 0 0 1 1 0 1 1 0 0 1 1 1 0
S = P | Q           1 0 1 0 0 1 0 0 0 1 0 1 0 0 0 1 0 0 1

M = (D|R|S)& !H& A  0 0 0 0 1 1 0 0 1 1 0 1 1 0 0 1 1 1 0 (final unit quads list)
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
D =     0 1 1 0 1 1 0
E =     1 0 0 0 0 0 0

H =     0 1 0 0 0 0 0

Third test

        1 1 1 1 1 1
        1 0 0 0 0 0
        1 0 0 0 0 0

A =     0 1 1 1 0 1
H =     0 0 0 0 0 0
M =     0 1 0 0 0 1
O =     0 0 1 1 0 0

Fourth test

        0 1 0 0 0 0
        0 0 0 0 0 0
        0 0 0 0 1 0

A =     1 1 1 1 1 1

H =     0 0 0 0 0 0

I =     0 0 0 0 0 0
J =     0 0 0 0 0 0
K =     0 0 0 1 0 1
L =     1 0 1 0 0 0
P =     0 1 0 0 0 0
Q =     0 0 0 0 1 0
R =     1 0 1 1 0 1
S =     0 1 0 0 1 0
M =     1 1 1 1 1 1
N =     0 0 0 0 0 0
O =     0 0 0 0 0 0

Out     U U U U U U

Counterexample candidate: fully exposed cap

Current Row         1 1 1 1 1 1 1

Depth -1 Rows
-1                  0 0 0 0 0 0 0
0                   0 0 0 0 0 0 0
+1                  0 0 0 0 0 0 0

A = CR - (CR ^ 0)   1 1 1 1 1 1 1 (opaque visibility of current row)

B = A & 0>>         0 0 0 0 0 0 0
C = A & 0<<         0 0 0 0 0 0 0
D = B | C           0 0 0 0 0 0 0

E = -1 & 0 & +1     0 0 0 0 0 0 0
F = A & E>>         0 0 0 0 0 0 0
G = A & E<<         0 0 0 0 0 0 0
H = (F|G)& !-1& !+1 0 0 0 0 0 0 0 (final v mergable quads list)

J = -1<< & -1>>     0 0 0 0 0 0 0
I = +1<< & +1>>     0 0 0 0 0 0 0
L = -1<< ^ -1>>     0 0 0 0 0 0 0
K = +1<< ^ +1>>     0 0 0 0 0 0 0
P = -1 & !J         0 0 0 0 0 0 0
Q = +1 & !I         0 0 0 0 0 0 0
R = K | L           0 0 0 0 0 0 0
S = P | Q           0 0 0 0 0 0 0

M = (D|R|S)& !H& A  0 0 0 0 0 0 0 (final unit quads list)
N = (I|J)& !H & !M  0 0 0 0 0 0 0
O = N & A           0 0 0 0 0 0 0 (final u mergable quads list)

Expected output     B B B B B B B

B = bidirectionally mergeable. All cells share the same AO signature here, so
the cap should merge like vanilla in both directions.


Abstract signature-only counterexample: non-flat signature with both right and down match

Interpretation:

- We have a 2x2 patch of visible cells.
- Three cells share the same non-flat AO signature.
- The fourth cell has a different AO signature.
- This is **not** currently backed by a concrete voxel arrangement. It is only a
  signature-level caution that `M / O / H` may need one more rule if such a
  geometry-backed case exists.

Visible-cell AO signatures:

                    x = 0          x = 1
z = 0               A              A
z = 1               A              B

A = `(3, 3, 2, 2)`
B = `(2, 2, 2, 1)`

So the top-left `A` cell matches:

- right neighbor: yes
- down neighbor: yes
- full 2x2 block: no, because the bottom-right cell is `B`

This is the important part:

- the `A` region is an L-shape of three cells
- an L-shape cannot be covered by one rectangle
- so the minimal rectangle cover here is **3 quads**, not 2

The ambiguity is not quad count. The ambiguity is **orientation**:

- valid partition 1: merge the top row `A A`, leave the lower-left `A` as a unit quad
- valid partition 2: merge the left column `A / A`, leave the upper-right `A` as a unit quad

In both cases the bottom-right `B` is a unit quad, so the total is 3 quads.

So if:

- `M` = unit only
- `O` = horizontal only
- `H` = vertical only
- `!(M|O|H)` = flat / vanilla

then this is still a problem for `M / O / H` by itself:

- the top-left `A` cell is not `M`
- it is not `O`-only
- it is not `H`-only
- it is also not flat / vanilla

It is a non-flat AO case that is locally mergeable in **either** direction, but
the final partition has to pick one direction consistently.

For now, the stronger evidence is still the geometry-backed third test above,
which exposed the missing internal-corner unit quad and was fixed by tightening
`H`.
