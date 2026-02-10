import QRCode from "react-qr-code";

export default function QR() {

  // builds correct URL for local + deployed
  const manualUrl = window.location.origin + "/manual";

  return (
    <div className="min-h-[70vh] flex flex-col items-center justify-center gap-8 p-10">

      <h1 className="text-3xl font-bold">
        Manual Page QR
      </h1>

      <div className="bg-white p-6 rounded-xl shadow">
        <QRCode value={manualUrl} size={220} />
      </div>

      <div className="text-center text-gray-600">
        Scan to open Manual page
      </div>

      <div className="text-sm text-gray-400 break-all text-center">
        {manualUrl}
      </div>

    </div>
  );
}
