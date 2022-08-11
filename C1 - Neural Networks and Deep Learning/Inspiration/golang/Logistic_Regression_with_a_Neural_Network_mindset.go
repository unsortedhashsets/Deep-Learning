package main

import (
	"fmt"
)

type LR struct {  
    infoMsg string
}

func (lr LR) info() {  
    fmt.Printf(lr.infoMsg)
}

func main() {
    lr := LR{"Here I will try to make golang program with the same features as Jupiter notebook has\n"}
	lr.info()
}