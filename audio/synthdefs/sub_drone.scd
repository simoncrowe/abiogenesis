// sub-drone: one near-static low sine/triangle drone anchoring the low end.
// Re-pitched rarely (scene changes); freq glides rather than stepping.
SynthDef("sub-drone", {
	arg out_bus = 0, freq = 55, amp = 0.2, attack = 8.0, release = 8.0, gate = 1;

	var f, gain, wob, sig, env;

	f = Lag.kr(freq, 5.0).clip(20, 220);
	gain = Lag.kr(amp, 1.0).clip(0, 1);

	// Very slight detune wobble keeps the drone alive without being audible as vibrato.
	wob = LFNoise2.kr(0.05).range(0.997, 1.003);
	sig = (SinOsc.ar(f * wob) * 0.8) + (LFTri.ar(f * wob * 2) * 0.12);
	sig = LPF.ar(sig, (f * 4).clip(80, 880));

	env = EnvGen.kr(Env.asr(attack.clip(0.01, 30), 1, release.clip(0.01, 30), \sin), gate, doneAction: 2);
	Out.ar(out_bus, Pan2.ar(sig, 0, gain * env));
})
