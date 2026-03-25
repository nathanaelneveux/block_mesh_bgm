Current Row     0 1 0 1 1 1 1 1 1 0 1 1 1 0 1 0

Depth -1 Rows
-1              1 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0
0               1 0 1 0 0 0 0 0 0 1 0 0 0 0 0 1
+1              1 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0


A = CR & 0>>    0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0
B = CR & 0<<    0 1 0 0 0 0 0 0 0 0 0 0 0 0 1 0
C = A & B       0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 (One self mergable type of AO signiture)
D = A ^ C       0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 (Another self mergable type of AO signiture)
E = B ^ C       0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 (Another self mergable type of AO signiture)

F = CR & -1     0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0
Find from F?    0 0 0 0 0 0 0 0 0 0 U U 0 0 0 0 (multiple unit quads)
G = CR & +1     0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0
Find from G?    0 0 0 0 U U M M U 0 0 0 0 0 0 0 (multiple unit quads + another self mergable type of AO signiture)


Another case    1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1

-1              0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 0
0               0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
+1              0 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0

Find            U U M M M U U Q Q U U N N N U U (multiple unit quads + remaining two self mergable AO signitures)