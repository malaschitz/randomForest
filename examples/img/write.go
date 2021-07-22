package img

import (
	"image"
	"image/png"
	"log"
	"os"
)

func WriteImage(data []byte, name string) {
	img := image.NewGray(image.Rect(0, 0, 28, 28))
	img.Pix = data
	out, err := os.Create("./" + name + ".png")
	if err != nil {
		log.Fatal(err)
	}
	err = png.Encode(out, img)
	if err != nil {
		log.Fatal(err)
	}
}
