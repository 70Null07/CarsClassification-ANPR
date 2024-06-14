using Compunet.YoloV8;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.IO;
using Tesseract;

namespace CarsClassification_ANPR
{
    public class CarClassification
    {
        public string ImgPath { get; set; }

        public CarClassification()
        {
            ImgPath = string.Empty;
        }
        public CarClassification(string _imgPath)
        {
            ImgPath = _imgPath;
        }

        public string GetClassPrediction()
        {
            if (ImgPath == string.Empty || !File.Exists(ImgPath))
                return string.Empty;

            ResizeImage(new Bitmap(ImgPath), 224, 224);

            using Image<Rgb24> image = SixLabors.ImageSharp.Image.Load<Rgb24>("./cropped_224_224.jpg");

            // We use DenseTensor for multi-dimensional access to populate the image data
            var mean = new[] { 0.485f, 0.456f, 0.406f };
            var stddev = new[] { 0.229f, 0.224f, 0.225f };
            DenseTensor<float> processedImage = new([1, 3, 224, 224]);
            image.ProcessPixelRows(accessor =>
            {
                for (int y = 0; y < accessor.Height; y++)
                {
                    Span<Rgb24> pixelSpan = accessor.GetRowSpan(y);
                    for (int x = 0; x < accessor.Width; x++)
                    {
                        processedImage[0, 0, y, x] = ((pixelSpan[x].R / 255f) - mean[0]) / stddev[0];
                        processedImage[0, 1, y, x] = ((pixelSpan[x].G / 255f) - mean[1]) / stddev[1];
                        processedImage[0, 2, y, x] = ((pixelSpan[x].B / 255f) - mean[2]) / stddev[2];
                    }
                }
            });

            // Pin tensor buffer and create a OrtValue with native tensor that makes use of
            // DenseTensor buffer directly. This avoids extra data copy within OnnxRuntime.
            // It will be unpinned on ortValue disposal
            using var inputOrtValue = OrtValue.CreateTensorValueFromMemory(OrtMemoryInfo.DefaultInstance,
                processedImage.Buffer, [1, 3, 224, 224]);

            var inputs = new Dictionary<string, OrtValue>
                {
                    { "input", inputOrtValue }
                };

            using var session = new InferenceSession("./carresnet152.onnx");
            using var runOptions = new RunOptions();
            using IDisposableReadOnlyCollection<OrtValue> results = session.Run(runOptions, inputs, session.OutputNames);

            // We copy results to array only to apply algorithms, otherwise data can be accessed directly
            // from the native buffer via ReadOnlySpan<T> or Span<T>
            var output = results[0].GetTensorDataAsSpan<float>().ToArray();
            float sum = output.Sum(x => (float)Math.Exp(x));
            IEnumerable<float> softmax = output.Select(x => (float)Math.Exp(x) / sum);

            IEnumerable<Prediction> prediction = softmax.Select((x, i) => new Prediction { Label = LabelMap.Labels[i], Confidence = x })
                                           .OrderByDescending(x => x.Confidence)
                                           .Take(1);

            if (prediction.Any())
                return prediction.First().Label;

            return string.Empty;
        }

        private static Bitmap ResizeImage(Bitmap image, int width, int height)
        {
            Bitmap resizedImage = new(width, height);
            using (Graphics graphics = Graphics.FromImage(resizedImage))
            {
                graphics.CompositingQuality = CompositingQuality.HighQuality;
                graphics.InterpolationMode = InterpolationMode.Bilinear;
                graphics.SmoothingMode = SmoothingMode.HighQuality;
                graphics.DrawImage(image, 0, 0, width, height);
            }

            resizedImage.Save($"./cropped_{width}_{height}.jpg");

            return resizedImage;
        }
    }

    public class NumberPlateRecognition
    {
        public string ImgPath { get; set; }
        private string? FrameImgPath { get; set; }

        public NumberPlateRecognition()
        {
            ImgPath = string.Empty;
            FrameImgPath = string.Empty;
        }
        public NumberPlateRecognition(string _imgPath)
        {
            ImgPath = _imgPath;
        }

        public string GetPrediction()
        {
            if (ImgPath == string.Empty || !File.Exists(ImgPath))
                return string.Empty;

            if (GetPlateFrame() && FrameImgPath != string.Empty)
            {
                using (var engine = new TesseractEngine(@"./testdata", "eng", EngineMode.Default))
                {
                    // Режим сегментации страницы
                    engine.DefaultPageSegMode = PageSegMode.SingleBlock;
                    // Маска разрешенных символов для российских номеров
                    engine.SetVariable("tessedit_char_whitelist", "ABEKMHOPCTYX0123456789");
                    using (var img = Pix.LoadFromFile(FrameImgPath))
                    {
                        using (var page = engine.Process(img))
                        {
                            string text = page.GetText();
                            text = text.Replace("\n", " ");

                            List<int> regions =
                            [
                                01, 02, 102, 702, 03, 04, 05, 06, 07, 08, 09, 10,
                                11, 12, 13, 113, 14, 15, 16, 116, 716, 17, 18, 19,
                                21, 121, 22, 23, 93, 123, 193, 24, 124, 25, 125, 26,
                                126, 27, 28, 29, 30, 31, 32, 33, 34, 134, 35, 36,
                                136, 37, 38, 138, 39, 40, 41, 42, 142, 43, 44, 45,
                                46, 47, 147, 48, 49, 50, 90, 150, 190, 750, 790, 51,
                                52, 152, 252, 53, 54, 154, 55, 155, 56, 156, 57, 58,
                                59, 159, 60, 61, 161, 761, 62, 63, 163, 763, 64, 164,
                                65, 66, 96, 196, 67, 68, 69, 70, 71, 72, 73, 173, 74,
                                174, 75, 76, 77, 97, 99, 177, 197, 199, 777, 797, 799, 977,
                                78, 98, 178, 198, 79, 80, 81, 82, 83, 84, 85, 86, 186,
                                87, 88, 94, 89, 92, 95
                            ];

                            if (text != string.Empty && text.Length > 4)
                            {
                                // Для 3 цифр
                                if (char.IsDigit(text[^2]) && char.IsDigit(text[^3]) && char.IsDigit(text[^4]))
                                {
                                    int plateRegion = int.Parse(text[^4].ToString() + text[^3] + text[^2]);
                                    if (regions.Contains(plateRegion))
                                        return text;
                                    else if (regions.Contains(int.Parse(text[^4].ToString() + text[^3])))
                                    {
                                        text = text.Remove(text.Length - 2, 1);
                                        return text;
                                    }
                                }
                            }

                            return text;
                        }
                    }
                }
            }

            return string.Empty;
        }

        private bool GetPlateFrame()
        {
            Bitmap CarImage = new(ImgPath);
            // Проверка на непустое изображение
            if (CarImage == null)
                return false;
            // Проверка на соответствие размерам модели
            if (CarImage.Width != 640 || CarImage.Height != 640)
                CarImage = ResizeImage(CarImage, 640, 640);


            using YoloV8Predictor predictor = YoloV8Predictor.Create("./plateyolov8.onnx");
            Compunet.YoloV8.Data.DetectionResult result = predictor.Detect("./cropped_640_640.jpg");

            if (result.Boxes.Length == 0)
                return false;
            else if (result.Boxes.Length == 1)
            {
                var currentBoxBounds = result.Boxes.First().Bounds;

                Bitmap croppedImage = new Bitmap(currentBoxBounds.Width, currentBoxBounds.Height,
                    System.Drawing.Imaging.PixelFormat.Format24bppRgb);

                using (Graphics graphics = Graphics.FromImage(croppedImage))
                {
                    graphics.DrawImage(CarImage, 0, 0, new System.Drawing.Rectangle(currentBoxBounds.Left,
                        currentBoxBounds.Top, currentBoxBounds.Width, currentBoxBounds.Height), GraphicsUnit.Pixel);
                }

                string savePath = $"./cropped_plate_{currentBoxBounds.Width}_{currentBoxBounds.Height}.jpg";

                croppedImage.Save(savePath);

                FrameImgPath = savePath;
            }
            else
            {
                throw new NotImplementedException("Не реализовано условие <число распознанных номеров> больше 1");
            }

            return true;
        }

        private static Bitmap ResizeImage(Bitmap image, int width, int height)
        {
            Bitmap resizedImage = new(width, height);
            using (Graphics graphics = Graphics.FromImage(resizedImage))
            {
                graphics.CompositingQuality = CompositingQuality.HighQuality;
                graphics.InterpolationMode = InterpolationMode.Bilinear;
                graphics.SmoothingMode = SmoothingMode.HighQuality;
                graphics.DrawImage(image, 0, 0, width, height);
            }

            resizedImage.Save($"./cropped_{width}_{height}.jpg");

            return resizedImage;
        }
    }

    internal class Prediction
    {
        public required string Label { get; set; }
        public float Confidence { get; set; }
    }

    public class LabelMap
    {
        public static readonly string[] Labels =
            [
            "AM General Hummer SUV 2000",
            "Acura Integra Type R 2001",
            "Acura RL Sedan 2012",
            "Acura TL Sedan 2012",
            "Acura TL Type-S 2008",
            "Acura TSX Sedan 2012",
            "Acura ZDX Hatchback 2012",
            "Aston Martin V8 Vantage Convertible 2012",
            "Aston Martin V8 Vantage Coupe 2012",
            "Aston Martin Virage Convertible 2012",
            "Aston Martin Virage Coupe 2012",
            "Audi 100 Sedan 1994",
            "Audi 100 Wagon 1994",
            "Audi A5 Coupe 2012",
            "Audi R8 Coupe 2012",
            "Audi RS 4 Convertible 2008",
            "Audi S4 Sedan 2007",
            "Audi S4 Sedan 2012",
            "Audi S5 Convertible 2012",
            "Audi S5 Coupe 2012",
            "Audi S6 Sedan 2011",
            "Audi TT Hatchback 2011",
            "Audi TT RS Coupe 2012",
            "Audi TTS Coupe 2012",
            "Audi V8 Sedan 1994",
            "BMW 1 Series Convertible 2012",
            "BMW 1 Series Coupe 2012",
            "BMW 3 Series Sedan 2012",
            "BMW 3 Series Wagon 2012",
            "BMW 6 Series Convertible 2007",
            "BMW ActiveHybrid 5 Sedan 2012",
            "BMW M3 Coupe 2012",
            "BMW M5 Sedan 2010",
            "BMW M6 Convertible 2010",
            "BMW X3 SUV 2012",
            "BMW X5 SUV 2007",
            "BMW X6 SUV 2012",
            "BMW Z4 Convertible 2012",
            "Bentley Arnage Sedan 2009",
            "Bentley Continental Flying Spur Sedan 2007",
            "Bentley Continental GT Coupe 2007",
            "Bentley Continental GT Coupe 2012",
            "Bentley Continental Supersports Conv. Convertible 2012",
            "Bentley Mulsanne Sedan 2011",
            "Bugatti Veyron 16.4 Convertible 2009",
            "Bugatti Veyron 16.4 Coupe 2009",
            "Buick Enclave SUV 2012",
            "Buick Rainier SUV 2007",
            "Buick Regal GS 2012",
            "Buick Verano Sedan 2012",
            "Cadillac CTS-V Sedan 2012",
            "Cadillac Escalade EXT Crew Cab 2007",
            "Cadillac SRX SUV 2012",
            "Chevrolet Avalanche Crew Cab 2012",
            "Chevrolet Camaro Convertible 2012",
            "Chevrolet Cobalt SS 2010",
            "Chevrolet Corvette Convertible 2012",
            "Chevrolet Corvette Ron Fellows Edition Z06 2007",
            "Chevrolet Corvette ZR1 2012",
            "Chevrolet Express Cargo Van 2007",
            "Chevrolet Express Van 2007",
            "Chevrolet HHR SS 2010",
            "Chevrolet Impala Sedan 2007",
            "Chevrolet Malibu Hybrid Sedan 2010",
            "Chevrolet Malibu Sedan 2007",
            "Chevrolet Monte Carlo Coupe 2007",
            "Chevrolet Silverado 1500 Classic Extended Cab 2007",
            "Chevrolet Silverado 1500 Extended Cab 2012",
            "Chevrolet Silverado 1500 Hybrid Crew Cab 2012",
            "Chevrolet Silverado 1500 Regular Cab 2012",
            "Chevrolet Silverado 2500HD Regular Cab 2012",
            "Chevrolet Sonic Sedan 2012",
            "Chevrolet Tahoe Hybrid SUV 2012",
            "Chevrolet TrailBlazer SS 2009",
            "Chevrolet Traverse SUV 2012",
            "Chrysler 300 SRT-8 2010",
            "Chrysler Aspen SUV 2009",
            "Chrysler Crossfire Convertible 2008",
            "Chrysler PT Cruiser Convertible 2008",
            "Chrysler Sebring Convertible 2010",
            "Chrysler Town and Country Minivan 2012",
            "Daewoo Nubira Wagon 2002",
            "Dodge Caliber Wagon 2007",
            "Dodge Caliber Wagon 2012",
            "Dodge Caravan Minivan 1997",
            "Dodge Challenger SRT8 2011",
            "Dodge Charger SRT-8 2009",
            "Dodge Charger Sedan 2012",
            "Dodge Dakota Club Cab 2007",
            "Dodge Dakota Crew Cab 2010",
            "Dodge Durango SUV 2007",
            "Dodge Durango SUV 2012",
            "Dodge Journey SUV 2012",
            "Dodge Magnum Wagon 2008",
            "Dodge Ram Pickup 3500 Crew Cab 2010",
            "Dodge Ram Pickup 3500 Quad Cab 2009",
            "Dodge Sprinter Cargo Van 2009",
            "Eagle Talon Hatchback 1998",
            "FIAT 500 Abarth 2012",
            "FIAT 500 Convertible 2012",
            "Ferrari 458 Italia Convertible 2012",
            "Ferrari 458 Italia Coupe 2012",
            "Ferrari California Convertible 2012",
            "Ferrari FF Coupe 2012",
            "Fisker Karma Sedan 2012",
            "Ford E-Series Wagon Van 2012",
            "Ford Edge SUV 2012",
            "Ford Expedition EL SUV 2009",
            "Ford F-150 Regular Cab 2007",
            "Ford F-150 Regular Cab 2012",
            "Ford F-450 Super Duty Crew Cab 2012",
            "Ford Fiesta Sedan 2012",
            "Ford Focus Sedan 2007",
            "Ford Freestar Minivan 2007",
            "Ford GT Coupe 2006",
            "Ford Mustang Convertible 2007",
            "Ford Ranger SuperCab 2011",
            "GMC Acadia SUV 2012",
            "GMC Canyon Extended Cab 2012",
            "GMC Savana Van 2012",
            "GMC Terrain SUV 2012",
            "GMC Yukon Hybrid SUV 2012",
            "Geo Metro Convertible 1993",
            "HUMMER H2 SUT Crew Cab 2009",
            "HUMMER H3T Crew Cab 2010",
            "Honda Accord Coupe 2012",
            "Honda Accord Sedan 2012",
            "Honda Odyssey Minivan 2007",
            "Honda Odyssey Minivan 2012",
            "Hyundai Accent Sedan 2012",
            "Hyundai Azera Sedan 2012",
            "Hyundai Elantra Sedan 2007",
            "Hyundai Elantra Touring Hatchback 2012",
            "Hyundai Genesis Sedan 2012",
            "Hyundai Santa Fe SUV 2012",
            "Hyundai Sonata Hybrid Sedan 2012",
            "Hyundai Sonata Sedan 2012",
            "Hyundai Tucson SUV 2012",
            "Hyundai Veloster Hatchback 2012",
            "Hyundai Veracruz SUV 2012",
            "Infiniti G Coupe IPL 2012",
            "Infiniti QX56 SUV 2011",
            "Isuzu Ascender SUV 2008",
            "Jaguar XK XKR 2012",
            "Jeep Compass SUV 2012",
            "Jeep Grand Cherokee SUV 2012",
            "Jeep Liberty SUV 2012",
            "Jeep Patriot SUV 2012",
            "Jeep Wrangler SUV 2012",
            "Lamborghini Aventador Coupe 2012",
            "Lamborghini Diablo Coupe 2001",
            "Lamborghini Gallardo LP 570-4 Superleggera 2012",
            "Lamborghini Reventon Coupe 2008",
            "Land Rover LR2 SUV 2012",
            "Land Rover Range Rover SUV 2012",
            "Lincoln Town Car Sedan 2011",
            "MINI Cooper Roadster Convertible 2012",
            "Maybach Landaulet Convertible 2012",
            "Mazda Tribute SUV 2011",
            "McLaren MP4-12C Coupe 2012",
            "Mercedes-Benz 300-Class Convertible 1993",
            "Mercedes-Benz C-Class Sedan 2012",
            "Mercedes-Benz E-Class Sedan 2012",
            "Mercedes-Benz S-Class Sedan 2012",
            "Mercedes-Benz SL-Class Coupe 2009",
            "Mercedes-Benz Sprinter Van 2012",
            "Mitsubishi Lancer Sedan 2012",
            "Nissan 240SX Coupe 1998",
            "Nissan Juke Hatchback 2012",
            "Nissan Leaf Hatchback 2012",
            "Nissan NV Passenger Van 2012",
            "Plymouth Neon Coupe 1999",
            "Porsche Panamera Sedan 2012",
            "Ram C-V Cargo Van Minivan 2012",
            "Rolls-Royce Ghost Sedan 2012",
            "Rolls-Royce Phantom Drophead Coupe Convertible 2012",
            "Rolls-Royce Phantom Sedan 2012",
            "Scion xD Hatchback 2012",
            "Spyker C8 Convertible 2009",
            "Spyker C8 Coupe 2009",
            "Suzuki Aerio Sedan 2007",
            "Suzuki Kizashi Sedan 2012",
            "Suzuki SX4 Hatchback 2012",
            "Suzuki SX4 Sedan 2012",
            "Tesla Model S Sedan 2012",
            "Toyota 4Runner SUV 2012",
            "Toyota Camry Sedan 2012",
            "Toyota Corolla Sedan 2012",
            "Toyota Gaia 2000",
            "Toyota Sequoia SUV 2012",
            "Volkswagen Beetle Hatchback 2012",
            "Volkswagen Golf Hatchback 1991",
            "Volkswagen Golf Hatchback 2012",
            "Volvo 240 Sedan 1993",
            "Volvo C30 Hatchback 2012",
            "Volvo XC90 SUV 2007",
            "smart fortwo Convertible 2012"];
    }

}
