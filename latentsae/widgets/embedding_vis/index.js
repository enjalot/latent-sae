
import React, { useRef, useEffect, useMemo } from "react";

import { extent } from 'https://esm.sh/d3-array@3.2.4';
import { scaleDiverging, scaleSequential } from 'https://esm.sh/d3-scale@4.0.2';
import { interpolateRdBu, interpolateCool } from 'https://esm.sh/d3-scale-chromatic@3.1.0';


export default function EmbeddingVisWidget({ embedding, rows, element_size, spacing, min_values, max_values }) {
  let padding = useMemo(() => 2*element_size, [element_size])
  let cols = useMemo(() => Math.floor(embedding.length / rows), [embedding, rows])
  let width = useMemo(() => element_size * cols + spacing * (cols - 1) + padding, [element_size, cols, spacing, padding])
  let height = useMemo(() => element_size * rows + spacing * (rows - 1) + padding, [element_size, rows, spacing, padding])

  useEffect(() => {
    if(!embedding || !embedding.length) return
    const ext = extent(embedding, d => d)
    const colorScale = scaleDiverging([ext[0], 0, ext[1]], interpolateRdBu)

    const canvas = container.current
    const ctx = canvas.getContext('2d')
    scaleCanvas(canvas, ctx, width, height)
    ctx.clearRect(0, 0, width, height)
    embedding.forEach((d,i) => {
      const x = (i % cols) * (element_size + spacing) + padding/2
      const y = Math.floor(i / cols) * (element_size + spacing) + padding/2
      let c = colorScale(d)
      if(min_values.length && max_values.length) {
        c = scaleDiverging([min_values[i], 0, max_values[i]], interpolateRdBu)(d)
      }
      ctx.fillStyle = c
      ctx.fillRect(x, y, element_size, element_size)
    })
  }, [embedding, rows, width, height, spacing, min_values, max_values, padding])

  const container = useRef();
  return (
    <div className="widget-embedding-vis">
      <canvas 
        ref={container} 
        width={width} 
        height={height} />
    </div>
  );
};




// https://gist.github.com/callumlocke/cc258a193839691f60dd
/**
 * This function takes a canvas, context, width and height. It scales both the
 * canvas and the context in such a way that everything you draw will be as
 * sharp as possible for the device.
 *
 * It doesn't return anything, it just modifies whatever canvas and context you
 * pass in.
 *
 * Adapted from Paul Lewis's code here:
 * http://www.html5rocks.com/en/tutorials/canvas/hidpi/
 */


function scaleCanvas(canvas, context, width, height) {
  // assume the device pixel ratio is 1 if the browser doesn't specify it
  const devicePixelRatio = window.devicePixelRatio || 1;

  // determine the 'backing store ratio' of the canvas context
  const backingStoreRatio = (
    context.webkitBackingStorePixelRatio ||
    context.mozBackingStorePixelRatio ||
    context.msBackingStorePixelRatio ||
    context.oBackingStorePixelRatio ||
    context.backingStorePixelRatio || 1
  );

  // determine the actual ratio we want to draw at
  const ratio = devicePixelRatio / backingStoreRatio;

  if (devicePixelRatio !== backingStoreRatio) {
    // set the 'real' canvas size to the higher width/height
    canvas.width = width * ratio;
    canvas.height = height * ratio;

    // ...then scale it back down with CSS
    canvas.style.width = width + 'px';
    canvas.style.height = height + 'px';
  }
  else {
    // this is a normal 1:1 device; just scale it simply
    canvas.width = width;
    canvas.height = height;
    canvas.style.width = '';
    canvas.style.height = '';
  }

  // scale the drawing context so everything will work at the higher ratio
  context.scale(ratio, ratio);
  context.imageSmoothingEnabled = false;

}