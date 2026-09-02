import { useEffect } from "react";
import { useRouter } from "next/router";

export default function YouthScoutingIndex() {
  const router = useRouter();

  useEffect(() => {
    router.replace("/youth-scouting/ranking");
  }, [router]);

  return null;
}
