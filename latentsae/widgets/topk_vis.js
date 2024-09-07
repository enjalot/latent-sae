
import * as React from "react";
// import { interpolateTurbo } from "d3-scale-chromatic";
import { interpolateTurbo } from "https://esm.sh/d3-scale-chromatic@3.1.0";


const ActivationBar = ({
  feature,
  activation,
  onHover = () => {},
  onSelect = () => {},
}) => {
  const featureColor = React.useMemo(() => interpolateTurbo(feature?.order), [feature])
  // const featureColor = "orange"
  return (
    <div className="sampleActivationBar" 
      onMouseEnter={() => onHover(feature)}
      onMouseLeave={() => onHover(null)}
      onClick={() => onSelect(feature)}
    >
      <div className="sampleActivationBarForeground"
        style={{
          width: `${activation/feature.max_activation * 100}%`, 
          backgroundColor: featureColor,
        }}
      >
      </div>
      <div className="sampleActivationBarLabel" 
      style={{
        // color: yiq(featureColor) >= 0.6 ? "#111" : "white",
      }}>
        <span>{feature.feature}: {feature.label}</span><span>{activation.toFixed(3)} ({(100*activation/feature.max_activation).toFixed(0)}%)</span>
      </div>
    </div>
  )
}

export default function TopkVisWidget({ features, data, n }) {
  console.log("DATA!!", data)
  console.log("FEATURES", features)
  return (
    <div className="widget-topk-vis">
      <div>
        {data.top_indices.slice(0, n).map((index, i) => {
          const feature = features[index]
          const activation = data.top_acts[i]
          return (
            <ActivationBar 
              key={i} 
              feature={feature} 
              activation={activation} 
              // onHover={(feature) => setData({ ...data, hoveredFeature: feature })}
              // onSelect={(feature) => setData({ ...data, selectedFeature: feature })}
            />
          )
        })}
      </div>
    </div>
  );
};

