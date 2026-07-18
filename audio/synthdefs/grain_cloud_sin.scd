// grain-cloud-sin: oscillator-grain "shimmer" voice (no buffer).
// The conductor keeps `freq` consonant by deriving it from the tonal anchor;
// `freqspread` supplies per-grain microtexture in semitones.
SynthDef("grain-cloud-sin", {
	arg out_bus = 0,
	freq = 440, freqspread = 2.0,
	density = 6, graindur = 0.15, durrand = 0.5,
	panwidth = 0.7, amp = 0.15,
	attack = 6.0, release = 6.0, gate = 1;

	var f, spread, dens, gdur, drnd, width, gain;
	var trig, dur, gfreq, gpan, env, sig;

	f = Lag.kr(freq, 1.0).clip(20, 8000);
	spread = Lag.kr(freqspread, 0.5).clip(0, 12);
	dens = Lag.kr(density, 0.5).clip(0.5, 80);
	gdur = Lag.kr(graindur, 0.5).clip(0.02, 0.8);
	drnd = Lag.kr(durrand, 0.5).clip(0, 1);
	width = Lag.kr(panwidth, 0.5).clip(0, 1);
	gain = Lag.kr(amp, 0.5).clip(0, 1);

	// Slow internal modulators so the cloud never freezes when the conductor is idle.
	dens = (dens * LFNoise2.kr(0.06).exprange(0.7, 1.4)).clip(0.5, 80);
	width = (width * LFNoise1.kr(0.05).range(0.7, 1.0)).clip(0, 1);

	trig = Dust.kr(dens);
	dur = gdur * (2 ** TRand.kr(drnd.neg, drnd, trig));
	// Bipolar spread in semitones around the centre frequency.
	gfreq = f * (2 ** (TRand.kr(spread.neg, spread, trig) / 12));
	gpan = TRand.kr(-1.0, 1.0, trig) * width;

	sig = GrainSin.ar(2, trig, dur, gfreq, gpan, -1, 512);
	env = EnvGen.kr(Env.asr(attack.clip(0.01, 30), 1, release.clip(0.01, 30), \sin), gate, doneAction: 2);
	Out.ar(out_bus, sig * gain * env);
})
