package rng

type RNG struct {
	state uint64
}

func NewRNG(seed uint64) *RNG {
	return &RNG{state: seed}
}

func (r *RNG) RandomU32() uint32 {
	r.state ^= (r.state >> 12)
	r.state ^= (r.state << 25)
	r.state ^= (r.state >> 27)
	return uint32((r.state * 0x2545F4914F6CDD1D) >> 32)
}

func (r *RNG) RandomFloat32() float32 {
	return float32(r.RandomU32()>>8) / 16777216.0
}

func (r *RNG) Uniform(a, b float32) float32 {
	return a + (b-a)*r.RandomFloat32()
}
