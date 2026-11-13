# AO merge mask examples

These cases trace the row classifier in [`src/ao.rs`](../src/ao.rs).
The three rows labeled `-1`, `0`, and `+1` lie in the plane just outside
the visible faces. A `1` marks an opaque voxel. The source row lies one
voxel behind row `0`. Its opaque cells are visible wherever row `0` is open.

The classifier names the main masks in code comments. The shifts below
move a row one bit. Complements apply within the row, and the classifier
limits its output to the query interior.

| Mask | Output | Meaning |
| --- | --- | --- |
| `H` | `N` | Merge across rows at width one |
| `M` | `U` | Emit a unit quad |
| `O` | `M` | Merge within the row |
| Remaining visible bits | `B` | Merge along both axes |
| No visible face | `0` | Emit nothing |

## 1. Mixed row

```text
Source row          1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1

Exterior plane
-1                  1 0 1 0 0 0 0 0 0 1 1 1 1 1 1 1 0 0 0
0                   1 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1
+1                  1 0 1 0 0 1 1 1 1 1 1 1 0 0 0 0 0 0 1

A = source & !0     0 1 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 (visible opaque faces)

B = A & 0>>         0 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 (a side neighbor is opaque)
C = A & 0<<         0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 (a side neighbor is opaque)
D = B | C           0 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 1 0 (either side is blocked)

E = -1 & 0 & +1     1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 (opaque through all three rows)
E*= !-1 & !0 & !+1  0 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 1 1 0 (open through all three rows)
F = E | E*          1 1 1 0 1 0 0 0 0 0 0 0 0 0 0 0 1 1 0 (uniform columns)
G =
(F<<&F>>)&(E<<|E>>) 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 (stable sides with opaque neighbor)
H = A& G& !-1& !+1  0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 (column-only mask)

J = -1<< & -1>>     0 1 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 (row merge candidate)
I = +1<< & +1>>     0 1 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 (row merge candidate)
L = -1<< ^ -1>>     0 0 0 1 1 1 0 0 0 0 0 1 1 0 0 0 0 1 0 (unit)
K = +1<< ^ +1>>     0 0 0 1 0 0 0 0 1 1 0 0 0 0 0 1 1 0 0 (unit)
P = -1 & !J         1 0 1 0 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 (possible unit)
Q = +1 & !I         1 0 1 0 0 1 0 0 0 0 0 1 0 0 0 0 0 0 1 (possible unit)
R = K | L           0 0 0 1 1 1 0 0 1 1 0 1 1 0 0 1 1 1 0
S = P | Q           1 0 1 0 0 1 0 0 0 1 0 1 0 0 0 1 0 0 1

M = (D|R|S)& !H& A  0 0 0 0 1 1 0 0 1 1 0 1 1 0 0 1 1 1 0 (unit-only mask)
N = (I|J)& !H & !M  0 0 0 0 0 0 1 1 0 0 1 0 0 1 1 0 0 0 0
O = N & A           0 0 0 0 0 0 1 1 0 0 1 0 0 1 1 0 0 0 0 (row-only mask)
Expected classes    0 N 0 0 U U M M U U M U U M M U U U 0
```

## 2. Continuous wall and isolated occluders

```text
        1 0 0 0 0 0 0
        1 0 0 1 0 0 1
        1 0 0 0 0 0 1

A =     0 1 1 0 1 1 0
B =     0 1 0 0 1 0 0
C =     0 0 1 0 0 1 0
D =     0 1 1 0 1 1 0
E =     1 0 0 0 0 0 0

H =     0 1 0 0 0 0 0
```

## 3. Inside corner

```text
        1 1 1 1 1 1
        1 0 0 0 0 0
        1 0 0 0 0 0

A =     0 1 1 1 0 1
H =     0 0 0 0 0 0
M =     0 1 0 0 0 1
O =     0 0 1 1 0 0
```

## 4. Offset occluders

```text
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

Class   U U U U U U
```

## 5. Asymmetric wall transition

```text
        1 0 1
        1 0 1
        1 0 1
        1 0 1
        1 0 0
        1 0 0

A =     1 1 1 1 1 1 (down the column)
H =     0 1 1 0 0 0 (down the column)
M =     1 0 0 1 1 1 (down the column)
O =     0 0 0 0 0 0 (down the column)

Class   U N N U U U (down the column)
```

The inside corner in case 3 stays a unit face. In case 5, a continuous
wall on one side allows a column merge only while the opposite side is
also uniform. The integration test
[`ao_safe_mode_splits_fifth_worked_example_in_every_direction`](compat.rs)
checks the transition for all face orientations.

## Fully exposed cap

When all three exterior rows are open, `H`, `M`, and `O` are zero.
Every visible cell can merge along both axes if its merge value matches.
The test `classify_marks_fully_exposed_cap_as_bidir` in [`src/ao.rs`](../src/ao.rs)
checks this case.

## Abstract AO signatures

Consider this hypothetical patch of corner-value tuples.

```text
A A       A = (3, 3, 2, 2)
A B       B = (2, 2, 2, 1)
```

The upper-left cell matches the cell to its right and the one below.
A rectangle can cover either matching pair. Both choices yield three quads
because the three `A` cells form an L and `B` needs its own quad. The
patch is a thought experiment in AO signatures without a corresponding
voxel fixture.
