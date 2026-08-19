"use client";

type DemoVideoProps = {
  title: string;
  poster: string;
};

export function DemoVideo({ title, poster }: DemoVideoProps) {
  return (
    <div className="demo__player">
      <video
        className="demo__video"
        controls
        playsInline
        preload="metadata"
        poster={poster}
        title={title}
      >
        <source src="/demo-video-1.mp4" type="video/mp4" />
      </video>
    </div>
  );
}
