// grain-cloud-buf: buffer-granulation texture voice.
//
// A single long-running synth node that generates its own grain stream with
// GrainBuf, triggered internally by Dust. The main thread only nudges the
// control-rate parameters below via /n_set a few times per second.
//
// All parameters are clamped in the SynthDef (mirrors the JS-side clamps) so a
// bad /n_set can never push a value out of range. Internal slow LFNoise
// modulators keep the texture alive even when the conductor is idle.
//
// out_bus is routed to BUS_SYNTH so the existing global FX chain
// (level -> lpf -> reverb) applies. The asr envelope with doneAction:2 lets the
// conductor cross-fade voices in/out (gate 1 -> attack, gate 0 -> release+free).

SynthDef('grain-cloud-buf', {
	|buf = 0, density = 10, graindur = 0.2, durrand = 0.3,
	 rate = 1, raterand = 0, pos = 0.5, posspray = 0.1, posdrift = 0.2,
	 panwidth = 0.5, amp = 0.2, attack = 4, release = 4, gate = 1, out_bus = 0|

	var dens, gdurC, durrandC, rateC, raterandC, posC, possprayC, panwidthC, ampC;
	var trig, gdur, prate, ppos, ppan, drift, wobble, env, sig;

	// Clamp to the documented ranges.
	dens      = density.clip(0.5, 80);
	gdurC     = graindur.clip(0.02, 0.8);
	durrandC  = durrand.clip(0, 1);
	rateC     = rate.clip(0.25, 4);
	raterandC = raterand.clip(0, 12);
	posC      = pos.clip(0, 1);
	possprayC = posspray.clip(0, 0.5);
	panwidthC = panwidth.clip(0, 1);
	ampC      = amp.clip(0, 1);

	// Internal slow modulators (0.02-0.2 Hz) so the cloud is never static.
	// posdrift steers how fast the read head wanders around `pos`.
	drift  = LFNoise2.kr(posdrift.clip(0, 1).linexp(0, 1, 0.02, 0.2)) * possprayC;
	wobble = 1 + (LFNoise1.kr(0.07) * 0.2); // gentle density breathing

	trig = Dust.kr(dens * wobble);

	// Per-grain random values, latched at each grain onset.
	gdur  = gdurC * (1 + TRand.kr(0, durrandC, trig));
	prate = rateC * (2 ** (TRand.kr(raterandC.neg, raterandC, trig) / 12));
	ppos  = (posC + drift + TRand.kr(possprayC.neg, possprayC, trig)).clip(0, 1);
	ppan  = TRand.kr(panwidthC.neg, panwidthC, trig);

	env = EnvGen.kr(Env.asr(attack.max(0), 1, release.max(0.01)), gate, doneAction: 2);

	// numChannels: 2 -> stereo grains; interp 2 -> linear; -1 default grain env.
	sig = GrainBuf.ar(2, trig, gdur, buf, prate, ppos, 2, ppan, -1);
	sig = sig * env * ampC;

	Out.ar(out_bus, sig);
}).writeDefFile(~outDir);
