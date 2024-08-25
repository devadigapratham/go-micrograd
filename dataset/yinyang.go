package dataset

import (
	"go-micrograd/rng"
	"math"
)

type Point struct {
	X, Y float32
	C    int
}

func GenDataYinYang(random *rng.RNG, n int) (train, val, test []Point) {
	var pts []Point

	//Setting helper functions for distances and class determination :

	distToRightDot := func(x, y, rBig float32) float32 {
		return float32(math.Sqrt(float64((x-1.5*rBig)*(x-1.5*rBig) + (y-rBig)*(y-rBig))))
	}

	distToLeftDot := func(x, y, rBig float32) float32 {
		return float32(math.Sqrt(float64((x-0.5*rBig)*(x-0.5*rBig) + (y-rBig)*(y-rBig))))
	}

	whichClass := func(x, y, rSmall, rBig float32) int {
		dRight := distToRightDot(x, y, rBig)
		dLeft := distToLeftDot(x, y, rBig)

		isYin := dRight <= rSmall || (dLeft > rSmall && dLeft <= 0.5*rBig) || (y > rBig && dRight > 0.5*rBig)
		isCircles := dRight < rSmall || dLeft < rSmall

		if isCircles {
			return 2
		}
		if isYin {
			return 0
		}
		return 1
	}

	// Data generation loop
	for i := 0; i < n; i++ {
		goalClass := i % 3
		for {
			x := random.Uniform(0, 2*0.5)
			y := random.Uniform(0, 2*0.5)

			if math.Sqrt(float64((x-0.5)*(x-0.5)+(y-0.5)*(y-0.5))) > 0.5 {
				continue
			}
			c := whichClass(x, y, 0.1, 0.5)
			if c == goalClass {
				pts = append(pts, Point{X: x, Y: y, C: c})
				break
			}
		}
	}

	// Split the dataset
	train = pts[:int(0.8*float32(n))]
	val = pts[int(0.8*float32(n)):int(0.9*float32(n))]
	test = pts[int(0.9*float32(n)):]

	return train, val, test

}
