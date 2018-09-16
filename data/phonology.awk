#!/usr/bin/awk -f

/^#/{
	print;
}
!/^#/{
#    printf("NR=%d\n", NR);
    for (i=1; i<=NF; i++) {
	if($i >= 0.4 ) {
	     printf("%s(%d) ",phon[i], i);
        }
    }
    printf("\n");
}

BEGIN{
# phon[1]-phon[23] : onset
# phon[24]-phon[37] : vowel
# phon[38]-phon[61] : coda
phon[1]="s";
phon[2]="S";
phon[3]="C";
phon[4]="z";
phon[5]="Z";
phon[6]="j";
phon[7]="f";
phon[8]="v";
phon[9]="T";
phon[10]="D";
phon[11]="p";
phon[12]="b";
phon[13]="t";
phon[14]="d";
phon[15]="k";
phon[16]="g";
phon[17]="m";
phon[18]="n";
phon[19]="h";
phon[20]="l";
phon[21]="r";
phon[22]="w";
phon[23]="y";
phon[24]="a";
phon[25]="e";
phon[26]="i";
phon[27]="o";
phon[28]="u";
phon[29]="@";
phon[30]="^";
phon[31]="A";
phon[32]="E";
phon[33]="I";
phon[34]="O";
phon[35]="U";
phon[36]="W";
phon[37]="Y";
phon[38]="r";
phon[39]="l";
phon[40]="m";
phon[41]="n";
phon[42]="N";
phon[43]="b";
phon[44]="g";
phon[45]="d";
phon[46]="ps";
phon[47]="ks";
phon[48]="ts";
phon[49]="s";
phon[50]="z";
phon[51]="f";
phon[52]="v";
phon[53]="p";
phon[54]="k";
phon[55]="t";
phon[56]="S";
phon[57]="Z";
phon[58]="T";
phon[59]="D";
phon[60]="C";
phon[61]="j";
}

