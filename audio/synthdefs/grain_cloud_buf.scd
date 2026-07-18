// grain-cloud-buf: buffer-granulation cloud voice.
// All control params are clamped here so bad /n_set values can't blow up.
// See architecture/15-granular-ambient.md for the parameter table.
SynthDef("grain-cloud-buf", {
	arg out_bus = 0, buf = 0,
	density = 8, graindur = 0.2, durrand = 0.5,
	rate = 1.0, raterand = 2.0,
	pos = 0.5, posspray = 0.1, posdrift = 0.2,
	panwidth = 0.5, amp = 0.2,
	attack = 6.0, release = 6.0, gate = 1;

	var dens, gdur, drnd, prate, prnd, cpos, spray, drift, width, gain;
	var trig, dur, grate, gpos, gpan, env, sig;

	// Smooth incoming /n_set steps (conductor sends a few times per second).
	dens = Lag.kr(density, 0.5).clip(0.5, 80);
	gdur = Lag.kr(graindur, 0.5).clip(0.02, 0.8);
	drnd = Lag.kr(durrand, 0.5).clip(0, 1);
	prate = Lag.kr(rate, 1.0).clip(0.25, 4);
	prnd = Lag.kr(raterand, 0.5).clip(0, 12);
	cpos = Lag.kr(pos, 1.0).clip(0, 1);
	spray = Lag.kr(posspray, 0.5).clip(0, 0.5);
	drift = Lag.kr(posdrift, 0.5).clip(0, 1);
	width = Lag.kr(panwidth, 0.5).clip(0, 1);
	gain = Lag.kr(amp, 0.5).clip(0, 1);

	// Slow internal modulators so the cloud never freezes when the conductor is idle.
	dens = (dens * LFNoise2.kr(0.07).exprange(0.7, 1.4)).clip(0.5, 80);
	width = (width * LFNoise1.kr(0.04).range(0.7, 1.0)).clip(0, 1);
	cpos = (cpos + (LFNoise2.kr(0.02 + (drift * 0.18)) * 0.2 * drift)).clip(0, 1);

	trig = Dust.kr(dens);
	// Grain duration multiplier in [2^-durrand, 2^durrand].
	dur = gdur * (2 ** TRand.kr(drnd.neg, drnd, trig));
	// Bipolar pitch jitter in semitones around the centre rate.
	grate = prate * (2 ** (TRand.kr(prnd.neg, prnd, trig) / 12));
	gpos = (cpos + TRand.kr(spray.neg, spray, trig)).clip(0, 1);
	gpan = TRand.kr(-1.0, 1.0, trig) * width;

	sig = GrainBuf.ar(2, trig, dur, buf, grate, gpos, 2, gpan, -1, 512);
	env = EnvGen.kr(Env.asr(attack.clip(0.01, 30), 1, release.clip(0.01, 30), \sin), gate, doneAction: 2);
	Out.ar(out_bus, sig * gain * env);
})
