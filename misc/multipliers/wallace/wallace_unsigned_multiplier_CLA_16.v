module wallace_unsigned_multiplier_CLA_16(product, A, B);
    input [15:0] A, B;
    output [31:0] product;

wire [15:0] pp0, pp1, pp2, pp3, pp4, pp5, pp6, pp7, pp8, pp9, pp10, pp11, pp12, pp13, pp14, pp15;

assign pp0[0] = A[0] & B[0];
assign pp0[1] = A[0] & B[1];
assign pp0[2] = A[0] & B[2];
assign pp0[3] = A[0] & B[3];
assign pp0[4] = A[0] & B[4];
assign pp0[5] = A[0] & B[5];
assign pp0[6] = A[0] & B[6];
assign pp0[7] = A[0] & B[7];
assign pp0[8] = A[0] & B[8];
assign pp0[9] = A[0] & B[9];
assign pp0[10] = A[0] & B[10];
assign pp0[11] = A[0] & B[11];
assign pp0[12] = A[0] & B[12];
assign pp0[13] = A[0] & B[13];
assign pp0[14] = A[0] & B[14];
assign pp0[15] = A[0] & B[15];
assign pp1[0] = A[1] & B[0];
assign pp1[1] = A[1] & B[1];
assign pp1[2] = A[1] & B[2];
assign pp1[3] = A[1] & B[3];
assign pp1[4] = A[1] & B[4];
assign pp1[5] = A[1] & B[5];
assign pp1[6] = A[1] & B[6];
assign pp1[7] = A[1] & B[7];
assign pp1[8] = A[1] & B[8];
assign pp1[9] = A[1] & B[9];
assign pp1[10] = A[1] & B[10];
assign pp1[11] = A[1] & B[11];
assign pp1[12] = A[1] & B[12];
assign pp1[13] = A[1] & B[13];
assign pp1[14] = A[1] & B[14];
assign pp1[15] = A[1] & B[15];
assign pp2[0] = A[2] & B[0];
assign pp2[1] = A[2] & B[1];
assign pp2[2] = A[2] & B[2];
assign pp2[3] = A[2] & B[3];
assign pp2[4] = A[2] & B[4];
assign pp2[5] = A[2] & B[5];
assign pp2[6] = A[2] & B[6];
assign pp2[7] = A[2] & B[7];
assign pp2[8] = A[2] & B[8];
assign pp2[9] = A[2] & B[9];
assign pp2[10] = A[2] & B[10];
assign pp2[11] = A[2] & B[11];
assign pp2[12] = A[2] & B[12];
assign pp2[13] = A[2] & B[13];
assign pp2[14] = A[2] & B[14];
assign pp2[15] = A[2] & B[15];
assign pp3[0] = A[3] & B[0];
assign pp3[1] = A[3] & B[1];
assign pp3[2] = A[3] & B[2];
assign pp3[3] = A[3] & B[3];
assign pp3[4] = A[3] & B[4];
assign pp3[5] = A[3] & B[5];
assign pp3[6] = A[3] & B[6];
assign pp3[7] = A[3] & B[7];
assign pp3[8] = A[3] & B[8];
assign pp3[9] = A[3] & B[9];
assign pp3[10] = A[3] & B[10];
assign pp3[11] = A[3] & B[11];
assign pp3[12] = A[3] & B[12];
assign pp3[13] = A[3] & B[13];
assign pp3[14] = A[3] & B[14];
assign pp3[15] = A[3] & B[15];
assign pp4[0] = A[4] & B[0];
assign pp4[1] = A[4] & B[1];
assign pp4[2] = A[4] & B[2];
assign pp4[3] = A[4] & B[3];
assign pp4[4] = A[4] & B[4];
assign pp4[5] = A[4] & B[5];
assign pp4[6] = A[4] & B[6];
assign pp4[7] = A[4] & B[7];
assign pp4[8] = A[4] & B[8];
assign pp4[9] = A[4] & B[9];
assign pp4[10] = A[4] & B[10];
assign pp4[11] = A[4] & B[11];
assign pp4[12] = A[4] & B[12];
assign pp4[13] = A[4] & B[13];
assign pp4[14] = A[4] & B[14];
assign pp4[15] = A[4] & B[15];
assign pp5[0] = A[5] & B[0];
assign pp5[1] = A[5] & B[1];
assign pp5[2] = A[5] & B[2];
assign pp5[3] = A[5] & B[3];
assign pp5[4] = A[5] & B[4];
assign pp5[5] = A[5] & B[5];
assign pp5[6] = A[5] & B[6];
assign pp5[7] = A[5] & B[7];
assign pp5[8] = A[5] & B[8];
assign pp5[9] = A[5] & B[9];
assign pp5[10] = A[5] & B[10];
assign pp5[11] = A[5] & B[11];
assign pp5[12] = A[5] & B[12];
assign pp5[13] = A[5] & B[13];
assign pp5[14] = A[5] & B[14];
assign pp5[15] = A[5] & B[15];
assign pp6[0] = A[6] & B[0];
assign pp6[1] = A[6] & B[1];
assign pp6[2] = A[6] & B[2];
assign pp6[3] = A[6] & B[3];
assign pp6[4] = A[6] & B[4];
assign pp6[5] = A[6] & B[5];
assign pp6[6] = A[6] & B[6];
assign pp6[7] = A[6] & B[7];
assign pp6[8] = A[6] & B[8];
assign pp6[9] = A[6] & B[9];
assign pp6[10] = A[6] & B[10];
assign pp6[11] = A[6] & B[11];
assign pp6[12] = A[6] & B[12];
assign pp6[13] = A[6] & B[13];
assign pp6[14] = A[6] & B[14];
assign pp6[15] = A[6] & B[15];
assign pp7[0] = A[7] & B[0];
assign pp7[1] = A[7] & B[1];
assign pp7[2] = A[7] & B[2];
assign pp7[3] = A[7] & B[3];
assign pp7[4] = A[7] & B[4];
assign pp7[5] = A[7] & B[5];
assign pp7[6] = A[7] & B[6];
assign pp7[7] = A[7] & B[7];
assign pp7[8] = A[7] & B[8];
assign pp7[9] = A[7] & B[9];
assign pp7[10] = A[7] & B[10];
assign pp7[11] = A[7] & B[11];
assign pp7[12] = A[7] & B[12];
assign pp7[13] = A[7] & B[13];
assign pp7[14] = A[7] & B[14];
assign pp7[15] = A[7] & B[15];
assign pp8[0] = A[8] & B[0];
assign pp8[1] = A[8] & B[1];
assign pp8[2] = A[8] & B[2];
assign pp8[3] = A[8] & B[3];
assign pp8[4] = A[8] & B[4];
assign pp8[5] = A[8] & B[5];
assign pp8[6] = A[8] & B[6];
assign pp8[7] = A[8] & B[7];
assign pp8[8] = A[8] & B[8];
assign pp8[9] = A[8] & B[9];
assign pp8[10] = A[8] & B[10];
assign pp8[11] = A[8] & B[11];
assign pp8[12] = A[8] & B[12];
assign pp8[13] = A[8] & B[13];
assign pp8[14] = A[8] & B[14];
assign pp8[15] = A[8] & B[15];
assign pp9[0] = A[9] & B[0];
assign pp9[1] = A[9] & B[1];
assign pp9[2] = A[9] & B[2];
assign pp9[3] = A[9] & B[3];
assign pp9[4] = A[9] & B[4];
assign pp9[5] = A[9] & B[5];
assign pp9[6] = A[9] & B[6];
assign pp9[7] = A[9] & B[7];
assign pp9[8] = A[9] & B[8];
assign pp9[9] = A[9] & B[9];
assign pp9[10] = A[9] & B[10];
assign pp9[11] = A[9] & B[11];
assign pp9[12] = A[9] & B[12];
assign pp9[13] = A[9] & B[13];
assign pp9[14] = A[9] & B[14];
assign pp9[15] = A[9] & B[15];
assign pp10[0] = A[10] & B[0];
assign pp10[1] = A[10] & B[1];
assign pp10[2] = A[10] & B[2];
assign pp10[3] = A[10] & B[3];
assign pp10[4] = A[10] & B[4];
assign pp10[5] = A[10] & B[5];
assign pp10[6] = A[10] & B[6];
assign pp10[7] = A[10] & B[7];
assign pp10[8] = A[10] & B[8];
assign pp10[9] = A[10] & B[9];
assign pp10[10] = A[10] & B[10];
assign pp10[11] = A[10] & B[11];
assign pp10[12] = A[10] & B[12];
assign pp10[13] = A[10] & B[13];
assign pp10[14] = A[10] & B[14];
assign pp10[15] = A[10] & B[15];
assign pp11[0] = A[11] & B[0];
assign pp11[1] = A[11] & B[1];
assign pp11[2] = A[11] & B[2];
assign pp11[3] = A[11] & B[3];
assign pp11[4] = A[11] & B[4];
assign pp11[5] = A[11] & B[5];
assign pp11[6] = A[11] & B[6];
assign pp11[7] = A[11] & B[7];
assign pp11[8] = A[11] & B[8];
assign pp11[9] = A[11] & B[9];
assign pp11[10] = A[11] & B[10];
assign pp11[11] = A[11] & B[11];
assign pp11[12] = A[11] & B[12];
assign pp11[13] = A[11] & B[13];
assign pp11[14] = A[11] & B[14];
assign pp11[15] = A[11] & B[15];
assign pp12[0] = A[12] & B[0];
assign pp12[1] = A[12] & B[1];
assign pp12[2] = A[12] & B[2];
assign pp12[3] = A[12] & B[3];
assign pp12[4] = A[12] & B[4];
assign pp12[5] = A[12] & B[5];
assign pp12[6] = A[12] & B[6];
assign pp12[7] = A[12] & B[7];
assign pp12[8] = A[12] & B[8];
assign pp12[9] = A[12] & B[9];
assign pp12[10] = A[12] & B[10];
assign pp12[11] = A[12] & B[11];
assign pp12[12] = A[12] & B[12];
assign pp12[13] = A[12] & B[13];
assign pp12[14] = A[12] & B[14];
assign pp12[15] = A[12] & B[15];
assign pp13[0] = A[13] & B[0];
assign pp13[1] = A[13] & B[1];
assign pp13[2] = A[13] & B[2];
assign pp13[3] = A[13] & B[3];
assign pp13[4] = A[13] & B[4];
assign pp13[5] = A[13] & B[5];
assign pp13[6] = A[13] & B[6];
assign pp13[7] = A[13] & B[7];
assign pp13[8] = A[13] & B[8];
assign pp13[9] = A[13] & B[9];
assign pp13[10] = A[13] & B[10];
assign pp13[11] = A[13] & B[11];
assign pp13[12] = A[13] & B[12];
assign pp13[13] = A[13] & B[13];
assign pp13[14] = A[13] & B[14];
assign pp13[15] = A[13] & B[15];
assign pp14[0] = A[14] & B[0];
assign pp14[1] = A[14] & B[1];
assign pp14[2] = A[14] & B[2];
assign pp14[3] = A[14] & B[3];
assign pp14[4] = A[14] & B[4];
assign pp14[5] = A[14] & B[5];
assign pp14[6] = A[14] & B[6];
assign pp14[7] = A[14] & B[7];
assign pp14[8] = A[14] & B[8];
assign pp14[9] = A[14] & B[9];
assign pp14[10] = A[14] & B[10];
assign pp14[11] = A[14] & B[11];
assign pp14[12] = A[14] & B[12];
assign pp14[13] = A[14] & B[13];
assign pp14[14] = A[14] & B[14];
assign pp14[15] = A[14] & B[15];
assign pp15[0] = A[15] & B[0];
assign pp15[1] = A[15] & B[1];
assign pp15[2] = A[15] & B[2];
assign pp15[3] = A[15] & B[3];
assign pp15[4] = A[15] & B[4];
assign pp15[5] = A[15] & B[5];
assign pp15[6] = A[15] & B[6];
assign pp15[7] = A[15] & B[7];
assign pp15[8] = A[15] & B[8];
assign pp15[9] = A[15] & B[9];
assign pp15[10] = A[15] & B[10];
assign pp15[11] = A[15] & B[11];
assign pp15[12] = A[15] & B[12];
assign pp15[13] = A[15] & B[13];
assign pp15[14] = A[15] & B[14];
assign pp15[15] = A[15] & B[15];

wire [276:0] S;
wire [276:0] Cout;

Half_Adder HA1 (pp0[1], pp1[0], S[0], Cout[0]);
Full_Adder FA2 (pp0[2], pp1[1], pp2[0], S[1], Cout[1]);
Full_Adder FA3 (pp0[3], pp1[2], pp2[1], S[2], Cout[2]);
Full_Adder FA4 (pp0[4], pp1[3], pp2[2], S[3], Cout[3]);
Half_Adder HA5 (pp3[1], pp4[0], S[4], Cout[4]);
Full_Adder FA6 (pp0[5], pp1[4], pp2[3], S[5], Cout[5]);
Full_Adder FA7 (pp3[2], pp4[1], pp5[0], S[6], Cout[6]);
Full_Adder FA8 (pp0[6], pp1[5], pp2[4], S[7], Cout[7]);
Full_Adder FA9 (pp3[3], pp4[2], pp5[1], S[8], Cout[8]);
Full_Adder FA10 (pp0[7], pp1[6], pp2[5], S[9], Cout[9]);
Full_Adder FA11 (pp3[4], pp4[3], pp5[2], S[10], Cout[10]);
Half_Adder HA12 (pp6[1], pp7[0], S[11], Cout[11]);
Full_Adder FA13 (pp0[8], pp1[7], pp2[6], S[12], Cout[12]);
Full_Adder FA14 (pp3[5], pp4[4], pp5[3], S[13], Cout[13]);
Full_Adder FA15 (pp6[2], pp7[1], pp8[0], S[14], Cout[14]);
Full_Adder FA16 (pp0[9], pp1[8], pp2[7], S[15], Cout[15]);
Full_Adder FA17 (pp3[6], pp4[5], pp5[4], S[16], Cout[16]);
Full_Adder FA18 (pp6[3], pp7[2], pp8[1], S[17], Cout[17]);
Full_Adder FA19 (pp0[10], pp1[9], pp2[8], S[18], Cout[18]);
Full_Adder FA20 (pp3[7], pp4[6], pp5[5], S[19], Cout[19]);
Full_Adder FA21 (pp6[4], pp7[3], pp8[2], S[20], Cout[20]);
Half_Adder HA22 (pp9[1], pp10[0], S[21], Cout[21]);
Full_Adder FA23 (pp0[11], pp1[10], pp2[9], S[22], Cout[22]);
Full_Adder FA24 (pp3[8], pp4[7], pp5[6], S[23], Cout[23]);
Full_Adder FA25 (pp6[5], pp7[4], pp8[3], S[24], Cout[24]);
Full_Adder FA26 (pp9[2], pp10[1], pp11[0], S[25], Cout[25]);
Full_Adder FA27 (pp0[12], pp1[11], pp2[10], S[26], Cout[26]);
Full_Adder FA28 (pp3[9], pp4[8], pp5[7], S[27], Cout[27]);
Full_Adder FA29 (pp6[6], pp7[5], pp8[4], S[28], Cout[28]);
Full_Adder FA30 (pp9[3], pp10[2], pp11[1], S[29], Cout[29]);
Full_Adder FA31 (pp0[13], pp1[12], pp2[11], S[30], Cout[30]);
Full_Adder FA32 (pp3[10], pp4[9], pp5[8], S[31], Cout[31]);
Full_Adder FA33 (pp6[7], pp7[6], pp8[5], S[32], Cout[32]);
Full_Adder FA34 (pp9[4], pp10[3], pp11[2], S[33], Cout[33]);
Half_Adder HA35 (pp12[1], pp13[0], S[34], Cout[34]);
Full_Adder FA36 (pp0[14], pp1[13], pp2[12], S[35], Cout[35]);
Full_Adder FA37 (pp3[11], pp4[10], pp5[9], S[36], Cout[36]);
Full_Adder FA38 (pp6[8], pp7[7], pp8[6], S[37], Cout[37]);
Full_Adder FA39 (pp9[5], pp10[4], pp11[3], S[38], Cout[38]);
Full_Adder FA40 (pp12[2], pp13[1], pp14[0], S[39], Cout[39]);
Full_Adder FA41 (pp0[15], pp1[14], pp2[13], S[40], Cout[40]);
Full_Adder FA42 (pp3[12], pp4[11], pp5[10], S[41], Cout[41]);
Full_Adder FA43 (pp6[9], pp7[8], pp8[7], S[42], Cout[42]);
Full_Adder FA44 (pp9[6], pp10[5], pp11[4], S[43], Cout[43]);
Full_Adder FA45 (pp12[3], pp13[2], pp14[1], S[44], Cout[44]);
Full_Adder FA46 (pp1[15], pp2[14], pp3[13], S[45], Cout[45]);
Full_Adder FA47 (pp4[12], pp5[11], pp6[10], S[46], Cout[46]);
Full_Adder FA48 (pp7[9], pp8[8], pp9[7], S[47], Cout[47]);
Full_Adder FA49 (pp10[6], pp11[5], pp12[4], S[48], Cout[48]);
Half_Adder HA50 (pp13[3], pp14[2], S[49], Cout[49]);
Full_Adder FA51 (pp2[15], pp3[14], pp4[13], S[50], Cout[50]);
Full_Adder FA52 (pp5[12], pp6[11], pp7[10], S[51], Cout[51]);
Full_Adder FA53 (pp8[9], pp9[8], pp10[7], S[52], Cout[52]);
Full_Adder FA54 (pp11[6], pp12[5], pp13[4], S[53], Cout[53]);
Full_Adder FA55 (pp3[15], pp4[14], pp5[13], S[54], Cout[54]);
Full_Adder FA56 (pp6[12], pp7[11], pp8[10], S[55], Cout[55]);
Full_Adder FA57 (pp9[9], pp10[8], pp11[7], S[56], Cout[56]);
Full_Adder FA58 (pp12[6], pp13[5], pp14[4], S[57], Cout[57]);
Full_Adder FA59 (pp4[15], pp5[14], pp6[13], S[58], Cout[58]);
Full_Adder FA60 (pp7[12], pp8[11], pp9[10], S[59], Cout[59]);
Full_Adder FA61 (pp10[9], pp11[8], pp12[7], S[60], Cout[60]);
Half_Adder HA62 (pp13[6], pp14[5], S[61], Cout[61]);
Full_Adder FA63 (pp5[15], pp6[14], pp7[13], S[62], Cout[62]);
Full_Adder FA64 (pp8[12], pp9[11], pp10[10], S[63], Cout[63]);
Full_Adder FA65 (pp11[9], pp12[8], pp13[7], S[64], Cout[64]);
Full_Adder FA66 (pp6[15], pp7[14], pp8[13], S[65], Cout[65]);
Full_Adder FA67 (pp9[12], pp10[11], pp11[10], S[66], Cout[66]);
Full_Adder FA68 (pp12[9], pp13[8], pp14[7], S[67], Cout[67]);
Full_Adder FA69 (pp7[15], pp8[14], pp9[13], S[68], Cout[68]);
Full_Adder FA70 (pp10[12], pp11[11], pp12[10], S[69], Cout[69]);
Half_Adder HA71 (pp13[9], pp14[8], S[70], Cout[70]);
Full_Adder FA72 (pp8[15], pp9[14], pp10[13], S[71], Cout[71]);
Full_Adder FA73 (pp11[12], pp12[11], pp13[10], S[72], Cout[72]);
Full_Adder FA74 (pp9[15], pp10[14], pp11[13], S[73], Cout[73]);
Full_Adder FA75 (pp12[12], pp13[11], pp14[10], S[74], Cout[74]);
Full_Adder FA76 (pp10[15], pp11[14], pp12[13], S[75], Cout[75]);
Half_Adder HA77 (pp13[12], pp14[11], S[76], Cout[76]);
Full_Adder FA78 (pp11[15], pp12[14], pp13[13], S[77], Cout[77]);
Full_Adder FA79 (pp12[15], pp13[14], pp14[13], S[78], Cout[78]);
Half_Adder HA80 (pp13[15], pp14[14], S[79], Cout[79]);
Half_Adder HA81 (Cout[0], S[1], S[80], Cout[80]);
Full_Adder FA82 (pp3[0], Cout[1], S[2], S[81], Cout[81]);
Full_Adder FA83 (Cout[2], S[3], S[4], S[82], Cout[82]);
Full_Adder FA84 (Cout[3], Cout[4], S[5], S[83], Cout[83]);
Full_Adder FA85 (pp6[0], Cout[5], Cout[6], S[84], Cout[84]);
Half_Adder HA86 (S[7], S[8], S[85], Cout[85]);
Full_Adder FA87 (Cout[7], Cout[8], S[9], S[86], Cout[86]);
Half_Adder HA88 (S[10], S[11], S[87], Cout[87]);
Full_Adder FA89 (Cout[9], Cout[10], Cout[11], S[88], Cout[88]);
Full_Adder FA90 (S[12], S[13], S[14], S[89], Cout[89]);
Full_Adder FA91 (pp9[0], Cout[12], Cout[13], S[90], Cout[90]);
Full_Adder FA92 (Cout[14], S[15], S[16], S[91], Cout[91]);
Full_Adder FA93 (Cout[15], Cout[16], Cout[17], S[92], Cout[92]);
Full_Adder FA94 (S[18], S[19], S[20], S[93], Cout[93]);
Full_Adder FA95 (Cout[18], Cout[19], Cout[20], S[94], Cout[94]);
Full_Adder FA96 (Cout[21], S[22], S[23], S[95], Cout[95]);
Half_Adder HA97 (S[24], S[25], S[96], Cout[96]);
Full_Adder FA98 (pp12[0], Cout[22], Cout[23], S[97], Cout[97]);
Full_Adder FA99 (Cout[24], Cout[25], S[26], S[98], Cout[98]);
Full_Adder FA100 (S[27], S[28], S[29], S[99], Cout[99]);
Full_Adder FA101 (Cout[26], Cout[27], Cout[28], S[100], Cout[100]);
Full_Adder FA102 (Cout[29], S[30], S[31], S[101], Cout[101]);
Full_Adder FA103 (S[32], S[33], S[34], S[102], Cout[102]);
Full_Adder FA104 (Cout[30], Cout[31], Cout[32], S[103], Cout[103]);
Full_Adder FA105 (Cout[33], Cout[34], S[35], S[104], Cout[104]);
Full_Adder FA106 (S[36], S[37], S[38], S[105], Cout[105]);
Full_Adder FA107 (pp15[0], Cout[35], Cout[36], S[106], Cout[106]);
Full_Adder FA108 (Cout[37], Cout[38], Cout[39], S[107], Cout[107]);
Full_Adder FA109 (S[40], S[41], S[42], S[108], Cout[108]);
Full_Adder FA110 (pp15[1], Cout[40], Cout[41], S[109], Cout[109]);
Full_Adder FA111 (Cout[42], Cout[43], Cout[44], S[110], Cout[110]);
Full_Adder FA112 (S[45], S[46], S[47], S[111], Cout[111]);
Full_Adder FA113 (pp14[3], pp15[2], Cout[45], S[112], Cout[112]);
Full_Adder FA114 (Cout[46], Cout[47], Cout[48], S[113], Cout[113]);
Full_Adder FA115 (Cout[49], S[50], S[51], S[114], Cout[114]);
Full_Adder FA116 (pp15[3], Cout[50], Cout[51], S[115], Cout[115]);
Full_Adder FA117 (Cout[52], Cout[53], S[54], S[116], Cout[116]);
Full_Adder FA118 (pp15[4], Cout[54], Cout[55], S[117], Cout[117]);
Full_Adder FA119 (Cout[56], Cout[57], S[58], S[118], Cout[118]);
Full_Adder FA120 (pp14[6], pp15[5], Cout[58], S[119], Cout[119]);
Full_Adder FA121 (Cout[59], Cout[60], Cout[61], S[120], Cout[120]);
Full_Adder FA122 (pp15[6], Cout[62], Cout[63], S[121], Cout[121]);
Half_Adder HA123 (Cout[64], S[65], S[122], Cout[122]);
Full_Adder FA124 (pp15[7], Cout[65], Cout[66], S[123], Cout[123]);
Half_Adder HA125 (Cout[67], S[68], S[124], Cout[124]);
Full_Adder FA126 (pp14[9], pp15[8], Cout[68], S[125], Cout[125]);
Half_Adder HA127 (Cout[69], Cout[70], S[126], Cout[126]);
Full_Adder FA128 (pp15[9], Cout[71], Cout[72], S[127], Cout[127]);
Full_Adder FA129 (pp15[10], Cout[73], Cout[74], S[128], Cout[128]);
Full_Adder FA130 (pp14[12], pp15[11], Cout[75], S[129], Cout[129]);
Half_Adder HA131 (Cout[80], S[81], S[130], Cout[130]);
Half_Adder HA132 (Cout[81], S[82], S[131], Cout[131]);
Full_Adder FA133 (S[6], Cout[82], S[83], S[132], Cout[132]);
Full_Adder FA134 (Cout[83], S[84], S[85], S[133], Cout[133]);
Full_Adder FA135 (Cout[84], Cout[85], S[86], S[134], Cout[134]);
Full_Adder FA136 (Cout[86], Cout[87], S[88], S[135], Cout[135]);
Full_Adder FA137 (S[17], Cout[88], Cout[89], S[136], Cout[136]);
Half_Adder HA138 (S[90], S[91], S[137], Cout[137]);
Full_Adder FA139 (S[21], Cout[90], Cout[91], S[138], Cout[138]);
Half_Adder HA140 (S[92], S[93], S[139], Cout[139]);
Full_Adder FA141 (Cout[92], Cout[93], S[94], S[140], Cout[140]);
Half_Adder HA142 (S[95], S[96], S[141], Cout[141]);
Full_Adder FA143 (Cout[94], Cout[95], Cout[96], S[142], Cout[142]);
Full_Adder FA144 (S[97], S[98], S[99], S[143], Cout[143]);
Full_Adder FA145 (Cout[97], Cout[98], Cout[99], S[144], Cout[144]);
Full_Adder FA146 (S[100], S[101], S[102], S[145], Cout[145]);
Full_Adder FA147 (S[39], Cout[100], Cout[101], S[146], Cout[146]);
Full_Adder FA148 (Cout[102], S[103], S[104], S[147], Cout[147]);
Full_Adder FA149 (S[43], S[44], Cout[103], S[148], Cout[148]);
Full_Adder FA150 (Cout[104], Cout[105], S[106], S[149], Cout[149]);
Full_Adder FA151 (S[48], S[49], Cout[106], S[150], Cout[150]);
Full_Adder FA152 (Cout[107], Cout[108], S[109], S[151], Cout[151]);
Full_Adder FA153 (S[52], S[53], Cout[109], S[152], Cout[152]);
Full_Adder FA154 (Cout[110], Cout[111], S[112], S[153], Cout[153]);
Full_Adder FA155 (S[55], S[56], S[57], S[154], Cout[154]);
Full_Adder FA156 (Cout[112], Cout[113], Cout[114], S[155], Cout[155]);
Full_Adder FA157 (S[59], S[60], S[61], S[156], Cout[156]);
Half_Adder HA158 (Cout[115], Cout[116], S[157], Cout[157]);
Full_Adder FA159 (S[62], S[63], S[64], S[158], Cout[158]);
Half_Adder HA160 (Cout[117], Cout[118], S[159], Cout[159]);
Full_Adder FA161 (S[66], S[67], Cout[119], S[160], Cout[160]);
Full_Adder FA162 (S[69], S[70], Cout[121], S[161], Cout[161]);
Full_Adder FA163 (S[71], S[72], Cout[123], S[162], Cout[162]);
Full_Adder FA164 (S[73], S[74], Cout[125], S[163], Cout[163]);
Half_Adder HA165 (S[75], S[76], S[164], Cout[164]);
Half_Adder HA166 (Cout[76], S[77], S[165], Cout[165]);
Half_Adder HA167 (pp15[12], Cout[77], S[166], Cout[166]);
Half_Adder HA168 (Cout[130], S[131], S[167], Cout[167]);
Half_Adder HA169 (Cout[131], S[132], S[168], Cout[168]);
Half_Adder HA170 (Cout[132], S[133], S[169], Cout[169]);
Full_Adder FA171 (S[87], Cout[133], S[134], S[170], Cout[170]);
Full_Adder FA172 (S[89], Cout[134], S[135], S[171], Cout[171]);
Full_Adder FA173 (Cout[135], S[136], S[137], S[172], Cout[172]);
Full_Adder FA174 (Cout[136], Cout[137], S[138], S[173], Cout[173]);
Full_Adder FA175 (Cout[138], Cout[139], S[140], S[174], Cout[174]);
Full_Adder FA176 (Cout[140], Cout[141], S[142], S[175], Cout[175]);
Full_Adder FA177 (Cout[142], Cout[143], S[144], S[176], Cout[176]);
Full_Adder FA178 (S[105], Cout[144], Cout[145], S[177], Cout[177]);
Half_Adder HA179 (S[146], S[147], S[178], Cout[178]);
Full_Adder FA180 (S[107], S[108], Cout[146], S[179], Cout[179]);
Full_Adder FA181 (Cout[147], S[148], S[149], S[180], Cout[180]);
Full_Adder FA182 (S[110], S[111], Cout[148], S[181], Cout[181]);
Full_Adder FA183 (Cout[149], S[150], S[151], S[182], Cout[182]);
Full_Adder FA184 (S[113], S[114], Cout[150], S[183], Cout[183]);
Full_Adder FA185 (Cout[151], S[152], S[153], S[184], Cout[184]);
Full_Adder FA186 (S[115], S[116], Cout[152], S[185], Cout[185]);
Full_Adder FA187 (Cout[153], S[154], S[155], S[186], Cout[186]);
Full_Adder FA188 (S[117], S[118], Cout[154], S[187], Cout[187]);
Full_Adder FA189 (Cout[155], S[156], S[157], S[188], Cout[188]);
Full_Adder FA190 (S[119], S[120], Cout[156], S[189], Cout[189]);
Full_Adder FA191 (Cout[157], S[158], S[159], S[190], Cout[190]);
Full_Adder FA192 (Cout[120], S[121], S[122], S[191], Cout[191]);
Full_Adder FA193 (Cout[158], Cout[159], S[160], S[192], Cout[192]);
Full_Adder FA194 (Cout[122], S[123], S[124], S[193], Cout[193]);
Half_Adder HA195 (Cout[160], S[161], S[194], Cout[194]);
Full_Adder FA196 (Cout[124], S[125], S[126], S[195], Cout[195]);
Half_Adder HA197 (Cout[161], S[162], S[196], Cout[196]);
Full_Adder FA198 (Cout[126], S[127], Cout[162], S[197], Cout[197]);
Full_Adder FA199 (Cout[127], S[128], Cout[163], S[198], Cout[198]);
Full_Adder FA200 (Cout[128], S[129], Cout[164], S[199], Cout[199]);
Full_Adder FA201 (S[78], Cout[129], Cout[165], S[200], Cout[200]);
Full_Adder FA202 (pp15[13], Cout[78], S[79], S[201], Cout[201]);
Full_Adder FA203 (pp14[15], pp15[14], Cout[79], S[202], Cout[202]);
Half_Adder HA204 (Cout[167], S[168], S[203], Cout[203]);
Half_Adder HA205 (Cout[168], S[169], S[204], Cout[204]);
Half_Adder HA206 (Cout[169], S[170], S[205], Cout[205]);
Half_Adder HA207 (Cout[170], S[171], S[206], Cout[206]);
Half_Adder HA208 (Cout[171], S[172], S[207], Cout[207]);
Full_Adder FA209 (S[139], Cout[172], S[173], S[208], Cout[208]);
Full_Adder FA210 (S[141], Cout[173], S[174], S[209], Cout[209]);
Full_Adder FA211 (S[143], Cout[174], S[175], S[210], Cout[210]);
Full_Adder FA212 (S[145], Cout[175], S[176], S[211], Cout[211]);
Full_Adder FA213 (Cout[176], S[177], S[178], S[212], Cout[212]);
Full_Adder FA214 (Cout[177], Cout[178], S[179], S[213], Cout[213]);
Full_Adder FA215 (Cout[179], Cout[180], S[181], S[214], Cout[214]);
Full_Adder FA216 (Cout[181], Cout[182], S[183], S[215], Cout[215]);
Full_Adder FA217 (Cout[183], Cout[184], S[185], S[216], Cout[216]);
Full_Adder FA218 (Cout[185], Cout[186], S[187], S[217], Cout[217]);
Full_Adder FA219 (Cout[187], Cout[188], S[189], S[218], Cout[218]);
Full_Adder FA220 (Cout[189], Cout[190], S[191], S[219], Cout[219]);
Full_Adder FA221 (Cout[191], Cout[192], S[193], S[220], Cout[220]);
Full_Adder FA222 (Cout[193], Cout[194], S[195], S[221], Cout[221]);
Full_Adder FA223 (S[163], Cout[195], Cout[196], S[222], Cout[222]);
Half_Adder HA224 (S[164], Cout[197], S[223], Cout[223]);
Half_Adder HA225 (S[165], Cout[198], S[224], Cout[224]);
Half_Adder HA226 (S[166], Cout[199], S[225], Cout[225]);
Half_Adder HA227 (Cout[166], Cout[200], S[226], Cout[226]);
Half_Adder HA228 (Cout[203], S[204], S[227], Cout[227]);
Half_Adder HA229 (Cout[204], S[205], S[228], Cout[228]);
Half_Adder HA230 (Cout[205], S[206], S[229], Cout[229]);
Half_Adder HA231 (Cout[206], S[207], S[230], Cout[230]);
Half_Adder HA232 (Cout[207], S[208], S[231], Cout[231]);
Half_Adder HA233 (Cout[208], S[209], S[232], Cout[232]);
Half_Adder HA234 (Cout[209], S[210], S[233], Cout[233]);
Half_Adder HA235 (Cout[210], S[211], S[234], Cout[234]);
Half_Adder HA236 (Cout[211], S[212], S[235], Cout[235]);
Full_Adder FA237 (S[180], Cout[212], S[213], S[236], Cout[236]);
Full_Adder FA238 (S[182], Cout[213], S[214], S[237], Cout[237]);
Full_Adder FA239 (S[184], Cout[214], S[215], S[238], Cout[238]);
Full_Adder FA240 (S[186], Cout[215], S[216], S[239], Cout[239]);
Full_Adder FA241 (S[188], Cout[216], S[217], S[240], Cout[240]);
Full_Adder FA242 (S[190], Cout[217], S[218], S[241], Cout[241]);
Full_Adder FA243 (S[192], Cout[218], S[219], S[242], Cout[242]);
Full_Adder FA244 (S[194], Cout[219], S[220], S[243], Cout[243]);
Full_Adder FA245 (S[196], Cout[220], S[221], S[244], Cout[244]);
Full_Adder FA246 (S[197], Cout[221], S[222], S[245], Cout[245]);
Full_Adder FA247 (S[198], Cout[222], S[223], S[246], Cout[246]);
Full_Adder FA248 (S[199], Cout[223], S[224], S[247], Cout[247]);
Full_Adder FA249 (S[200], Cout[224], S[225], S[248], Cout[248]);
Full_Adder FA250 (S[201], Cout[225], S[226], S[249], Cout[249]);
Full_Adder FA251 (Cout[201], S[202], Cout[226], S[250], Cout[250]);
Half_Adder HA252 (pp15[15], Cout[202], S[251], Cout[251]);



/* Final Stage */
wire [24:0] G, P, C;
assign G[0]  = Cout[227] & S[228];
assign G[1]  = Cout[228] & S[229];
assign G[2]  = Cout[229] & S[230];
assign G[3]  = Cout[230] & S[231];
assign G[4]  = Cout[231] & S[232];
assign G[5]  = Cout[232] & S[233];
assign G[6]  = Cout[233] & S[234];
assign G[7]  = Cout[234] & S[235];
assign G[8]  = Cout[235] & S[236];
assign G[9]  = Cout[236] & S[237];
assign G[10] = Cout[237] & S[238];
assign G[11] = Cout[238] & S[239];
assign G[12] = Cout[239] & S[240];
assign G[13] = Cout[240] & S[241];
assign G[14] = Cout[241] & S[242];
assign G[15] = Cout[242] & S[243];
assign G[16] = Cout[243] & S[244];
assign G[17] = Cout[244] & S[245];
assign G[18] = Cout[245] & S[246];
assign G[19] = Cout[246] & S[247];
assign G[20] = Cout[247] & S[248];
assign G[21] = Cout[248] & S[249];
assign G[22] = Cout[249] & S[250];
assign G[23] = Cout[250] & S[251];
assign G[24] = Cout[251] & 0;
assign P[0]  = Cout[227] ^ S[228];
assign P[1]  = Cout[228] ^ S[229];
assign P[2]  = Cout[229] ^ S[230];
assign P[3]  = Cout[230] ^ S[231];
assign P[4]  = Cout[231] ^ S[232];
assign P[5]  = Cout[232] ^ S[233];
assign P[6]  = Cout[233] ^ S[234];
assign P[7]  = Cout[234] ^ S[235];
assign P[8]  = Cout[235] ^ S[236];
assign P[9]  = Cout[236] ^ S[237];
assign P[10] = Cout[237] ^ S[238];
assign P[11] = Cout[238] ^ S[239];
assign P[12] = Cout[239] ^ S[240];
assign P[13] = Cout[240] ^ S[241];
assign P[14] = Cout[241] ^ S[242];
assign P[15] = Cout[242] ^ S[243];
assign P[16] = Cout[243] ^ S[244];
assign P[17] = Cout[244] ^ S[245];
assign P[18] = Cout[245] ^ S[246];
assign P[19] = Cout[246] ^ S[247];
assign P[20] = Cout[247] ^ S[248];
assign P[21] = Cout[248] ^ S[249];
assign P[22] = Cout[249] ^ S[250];
assign P[23] = Cout[250] ^ S[251];
assign P[24] = Cout[251] ^ 0;

assign C[0] = 0;
assign C[1] = G[0] | (P[0] & C[0]);
assign C[2] = G[1] | (P[1] & C[1]);
assign C[3] = G[2] | (P[2] & C[2]);
assign C[4] = G[3] | (P[3] & C[3]);
assign C[5] = G[4] | (P[4] & C[4]);
assign C[6] = G[5] | (P[5] & C[5]);
assign C[7] = G[6] | (P[6] & C[6]);
assign C[8] = G[7] | (P[7] & C[7]);
assign C[9] = G[8] | (P[8] & C[8]);
assign C[10] = G[9] | (P[9] & C[9]);
assign C[11] = G[10] | (P[10] & C[10]);
assign C[12] = G[11] | (P[11] & C[11]);
assign C[13] = G[12] | (P[12] & C[12]);
assign C[14] = G[13] | (P[13] & C[13]);
assign C[15] = G[14] | (P[14] & C[14]);
assign C[16] = G[15] | (P[15] & C[15]);
assign C[17] = G[16] | (P[16] & C[16]);
assign C[18] = G[17] | (P[17] & C[17]);
assign C[19] = G[18] | (P[18] & C[18]);
assign C[20] = G[19] | (P[19] & C[19]);
assign C[21] = G[20] | (P[20] & C[20]);
assign C[22] = G[21] | (P[21] & C[21]);
assign C[23] = G[22] | (P[22] & C[22]);
assign C[24] = G[23] | (P[23] & C[23]);
assign cout  = G[24] | (P[24] & C[24]);

assign product[7] = P[0];
assign product[8] = P[1] ^ C[1];
assign product[9] = P[2] ^ C[2];
assign product[10] = P[3] ^ C[3];
assign product[11] = P[4] ^ C[4];
assign product[12] = P[5] ^ C[5];
assign product[13] = P[6] ^ C[6];
assign product[14] = P[7] ^ C[7];
assign product[15] = P[8] ^ C[8];
assign product[16] = P[9] ^ C[9];
assign product[17] = P[10] ^ C[10];
assign product[18] = P[11] ^ C[11];
assign product[19] = P[12] ^ C[12];
assign product[20] = P[13] ^ C[13];
assign product[21] = P[14] ^ C[14];
assign product[22] = P[15] ^ C[15];
assign product[23] = P[16] ^ C[16];
assign product[24] = P[17] ^ C[17];
assign product[25] = P[18] ^ C[18];
assign product[26] = P[19] ^ C[19];
assign product[27] = P[20] ^ C[20];
assign product[28] = P[21] ^ C[21];
assign product[29] = P[22] ^ C[22];
assign product[30] = P[23] ^ C[23];
assign product[31] = P[24] ^ C[24];




assign product[6] = S[227];
assign product[5] = S[203];
assign product[4] = S[167];
assign product[3] = S[130];
assign product[2] = S[80];
assign product[1] = S[0];
assign product[0] = pp0[0];
endmodule

module Half_Adder(input wire in1,
                  input wire in2,
                  output wire sum,
                  output wire cout);
    xor(sum, in1, in2);
    and(cout, in1, in2);
endmodule

module Full_Adder(input wire in1,
                  input wire in2,
                  input wire cin,
                  output wire sum,
                  output wire cout);
    wire temp1;
    wire temp2;
    wire temp3;
    xor(sum, in1, in2, cin);
    and(temp1,in1,in2);
    and(temp2,in1,cin);
    and(temp3,in2,cin);
    or(cout,temp1,temp2,temp3);
endmodule