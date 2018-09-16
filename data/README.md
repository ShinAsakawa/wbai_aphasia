---
title: About PMSP96 data
author: Shin Asakawa asakawa@ieee.org
date: 01/Sep/2018
---

# Data description

このディレクトリには PMSP96 のデータが納められている。
convert.sed がオリジナルデータ SM-nsyl.ex に対して
sed -f convert.sed SM-nsyl.ex したのが PMSP96.orig である。

```bash
sed -f convert.sed SM-nsyl.ex > PMSP96.orig
sed -f convert.sed SM-nsyl.ex | grep '^#' | sed -e 's/# [0-9]* //g' | sed -e 's/ [0-9]\.[0-9]* .*$//g' > pmsp96.data
cat pmsp96.data | sed -e 's/ .*$//g' > pmsp96.input
cat pmsp96.data | sed -e 's/^.* //g' > pmsp96.teacher
```

PMSP96.orig は、1 行目がコメント、2 行目が orthography data, 3 行目が
phonology データ、となっていて 3 行で 1 データになっている。
awk の剰余演算子を使って PMSP96.input, PMSP96.teach が出来上がっている。

その他の sed scripts は hme2 のテスト用として開発した。
extract_G1.sed は hme2 の吐き出した標準エラー出力をリダイレクトした
ファイルを引数にして top gate の出力を抽出する。
extract_G2.sed は hme2 の吐き出した標準エラー出力をリダイレクトした
ファイルを引数にして second gate の出力を抽出する。
extract_output.sed は hme2 の吐き出した標準エラー出力をリダイレクトした
ファイルを引数にして出力信号を抽出する。

extract_output によって出力を取り出し、`phonology.awk` スクリプトに掛けると
実際の読みを出力する。

judge scripts は正解率を求める scripts である。judge は内部で PMSP96.teach
を読み込んで正解データとしているため PMSP96.teach は消してはならない。

onset: Y S P T K Q C B D G F V J Z L M N R W H CH GH GN PH PS RH SH TH TS WH: 30
vowel: E I O U A Y Al AU AW AY EA EE El EU EW EY IE OA OE Ol OO OU OW OY UE Ul UY: 27
coda: H R L M N B D G C X F V ∫ S Z P T K Q BB CH CK DD DG FF GG GH GN KS LL NG NN PH PP PS RR SH SL SS TCH TH TS TT ZZ U E ES ED: 48


# Table 2
Phonological and Orthographic Representations Used in the Simulations

Phonology
----------+-------------------------------------------------
onset       s S C z Z j f v T D p b t d k g m n h I r w y
vowel       a e i o u @ ^ A E I O U W Y
coda        r I m n N b g d ps ks ts s z f v p k t S Z T D C j

Orthography
------------+----------------------------------------------------------------
onset Y S P T K Q C B D G F V J Z LM N R W H CH GH GN PH PS RH SH TH TS WH
vowel E I O U A Y AI AU AW AY EA EE EI EU EW EY IE OA OE OI OO OU OW OY UE UI UY
coda H R L M N B D G C X F V ∫ S Z P T K Q BB CH CK DD DG FF GG GH GN KS LL NG NN PH PP PS RR SH SL SS TCH TH TS TT ZZ U E ES ED

Note. The notation for vowels is slightly different from that used by
Seidenberg and McClelland (1989). Also, the representations differ
slightly from those: used by Plaut and McClelland (1993; Seidenberg,
Plaut, Petersen, McClelland, & McRae, 1994). In particular, /C/ and
/j/ has been added for /tS/ and /dZ/, the ordering of phonemes is
somewhat different, the mutually exclusive phoneme sets have been
added, and the consonantal graphemes U, GU, and QU have been
eliminated. These changes captures the relevant phonotac¬tic
constraints better and simplify the encoding procedure for converting
letter strings into activity patterns over grapheme units. /a/ in POT,
/@/ in CAT, /e/ in BED, /i/ in HIT, /0/ in DOG, /u/ in GOOD, /A/ in
Make, /E/ in KEEP, /I/ in BIKE, /O/ in HOPE, /U/ in BOOT, /W/ in NOW,
/Y/ in BOY, /^/ in CUP, /N/ in RING, /S/ in SHE, /C/ in CHIN, /Z/ in
BEIGE, /T/ in THIN, /D/ in THIS. All other phonemes are represented in
the conventional way (e.g. /b/ in BAT).  The groupings indicate sets
of mutually exclusive phonemes.

- Number of grapheme units:  105
- Number of phoneme units 61 (onset=23, vowel=14, coda=24, total=23+14+24=61)
