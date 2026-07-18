// grain-cloud-sin: "shimmer" voice using GrainSin (pitched sine-blip grains).
//
// No buffer: grains are short sine bursts around a centre frequency. Consonant
// transposition is the conductor's job (it picks `freq` from the tonal anchor
// times a just-intonation ratio); freqspread supplies per-grain microtexture.
//
// Same density/duration/pan/amp/envelope scheme as grain-cloud-buf.

SynthDef('grain-cloud-sin', {
	|density = 8, graindur = 0.15, durrand = 0.3,
	 freq = 220, freqspread = 0, panwidth = 0.6,
	 amp = 0.15, attack = 5, release = 5, gate = 1, out_bus = 0|

	var dens, gdurC, durrandC, freqC, freqspreadC, panwidthC, ampC;
	var trig, gdur, gfreq, ppan, wobble, env, sig;

	dens        = density.clip(0.5, 80);
	gdurC       = graindur.clip(0.02, 0.8);
	durrandC    = durrand.clip(0, 1);
	freqC       = freq.clip(20, 12000);
	freqspreadC = freqspread.clip(0, 12);
	panwidthC   = panwidth.clip(0, 1);
	ampC        = amp.clip(0, 1);

	wobble = 1 + (LFNoise1.kr(0.05) * 0.2);
	trig = Dust.kr(dens * wobble);

	gdur  = gdurC * (1 + TRand.kr(0, durrandC, trig));
	gfreq = freqC * (2 ** (TRand.kr(freqspreadC.neg, freqspreadC, trig) / 12));
	ppan  = TRand.kr(panwidthC.neg, panwidthC, trig);

	env = EnvGen.kr(Env.asr(attack.max(0), 1, release.max(0.01)), gate, doneAction: 2);

	sig = GrainSin.ar(2, trig, gdur, gfreq, ppan);
	sig = sig * env * ampC;

	Out.ar(out_bus, sig);
}).writeDefFile(~outDir);
