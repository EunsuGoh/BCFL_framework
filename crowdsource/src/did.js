"use strict";
var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
var __generator = (this && this.__generator) || function (thisArg, body) {
    var _ = { label: 0, sent: function() { if (t[0] & 1) throw t[1]; return t[1]; }, trys: [], ops: [] }, f, y, t, g;
    return g = { next: verb(0), "throw": verb(1), "return": verb(2) }, typeof Symbol === "function" && (g[Symbol.iterator] = function() { return this; }), g;
    function verb(n) { return function (v) { return step([n, v]); }; }
    function step(op) {
        if (f) throw new TypeError("Generator is already executing.");
        while (g && (g = 0, op[0] && (_ = 0)), _) try {
            if (f = 1, y && (t = op[0] & 2 ? y["return"] : op[0] ? y["throw"] || ((t = y["return"]) && t.call(y), 0) : y.next) && !(t = t.call(y, op[1])).done) return t;
            if (y = 0, t) op = [op[0] & 2, t.value];
            switch (op[0]) {
                case 0: case 1: t = op; break;
                case 4: _.label++; return { value: op[1], done: false };
                case 5: _.label++; y = op[1]; op = [0]; continue;
                case 7: op = _.ops.pop(); _.trys.pop(); continue;
                default:
                    if (!(t = _.trys, t = t.length > 0 && t[t.length - 1]) && (op[0] === 6 || op[0] === 2)) { _ = 0; continue; }
                    if (op[0] === 3 && (!t || (op[1] > t[0] && op[1] < t[3]))) { _.label = op[1]; break; }
                    if (op[0] === 6 && _.label < t[1]) { _.label = t[1]; t = op; break; }
                    if (t && _.label < t[2]) { _.label = t[2]; _.ops.push(op); break; }
                    if (t[2]) _.ops.pop();
                    _.trys.pop(); continue;
            }
            op = body.call(thisArg, _);
        } catch (e) { op = [6, e]; y = 0; } finally { f = t = 0; }
        if (op[0] & 5) throw op[1]; return { value: op[0] ? op[1] : void 0, done: true };
    }
};
exports.__esModule = true;
// import { Resolver } from "@ethersproject/providers";
require("dotenv/config");
var did_jwt_vc_1 = require("did-jwt-vc");
var Issuer = require("did-jwt-vc").Issuer;
var ethers = require("ethers");
var EthrDID = require("ethr-did").EthrDID;
var Wallet = require("@ethersproject/wallet").Wallet;
var Web3 = require('web3');
var did_resolver_1 = require("did-resolver");
var ethr_did_resolver_1 = require("ethr-did-resolver");
// const provider = new Web3.providers.HttpProvider("http://127.0.0.1:7545")
// const web3 = new Web3(provider);
var rpcURL = "http://127.0.0.1:7545"; // ganache
var provider = new ethers.providers.JsonRpcProvider(rpcURL);
var didContractAdd = process.env.DID_CONTRACT_ADDRESS;
var createIssuerDID = function (issuerPub, issuerPriv) { return __awaiter(void 0, void 0, void 0, function () {
    var txSigner, rpcUrl, issuer;
    return __generator(this, function (_a) {
        txSigner = new Wallet(issuerPriv, provider);
        try {
            rpcUrl = process.env.RPC_URL;
            issuer = new EthrDID({
                txSigner: txSigner,
                provider: provider,
                identifier: issuerPub,
                privateKey: issuerPriv,
                rpcUrl: rpcUrl,
                chainNameOrId: "ganache",
                registry: didContractAdd
            });
            console.log(issuer.did);
            return [2 /*return*/, issuer];
        }
        catch (e) {
            console.log(e);
        }
        return [2 /*return*/];
    });
}); };
var createHolderDID = function (holderPub, holderPriv) { return __awaiter(void 0, void 0, void 0, function () {
    var txSigner, rpcUrl, holder;
    return __generator(this, function (_a) {
        txSigner = new Wallet(holderPriv, provider);
        try {
            rpcUrl = process.env.RPC_URL;
            holder = new EthrDID({
                txSigner: txSigner,
                provider: provider,
                identifier: holderPub,
                privateKey: holderPriv,
                rpcUrl: rpcUrl,
                chainNameOrId: "ganache",
                registry: didContractAdd
            });
            console.log(holder.did);
            return [2 /*return*/, holder.did];
        }
        catch (e) {
            console.log(e);
        }
        return [2 /*return*/];
    });
}); };
var issueVC = function (did, issuerPub, issuerPriv) { return __awaiter(void 0, void 0, void 0, function () {
    var issuer, vcPayload, vcJwt;
    return __generator(this, function (_a) {
        switch (_a.label) {
            case 0: return [4 /*yield*/, createIssuerDID(issuerPub, issuerPriv)];
            case 1:
                issuer = _a.sent();
                vcPayload = {
                    sub: "did:ethr:ganache:" + did,
                    nbf: 1562950282,
                    vc: {
                        "@context": ["https://www.w3.org/2018/credentials/v1"],
                        type: ["VerifiableCredential"],
                        credentialSubject: {
                            // GOVERN_FA_PASSPORT Inner join GOVERN_USER_CLIENT
                            trainer: {
                                did: did
                            }
                        }
                    }
                };
                return [4 /*yield*/, (0, did_jwt_vc_1.createVerifiableCredentialJwt)(vcPayload, issuer)];
            case 2:
                vcJwt = _a.sent();
                console.log("VCJWT::" + vcJwt);
                return [2 /*return*/, vcJwt];
        }
    });
}); };
var verifyVC = function (vc, issuerDid) { return __awaiter(void 0, void 0, void 0, function () {
    var providerConfig, ethrDidResolver, didResolver, verifiedVc, e_1;
    return __generator(this, function (_a) {
        switch (_a.label) {
            case 0:
                _a.trys.push([0, 2, , 3]);
                providerConfig = {
                    name: "ganache",
                    rpcUrl: rpcURL,
                    registry: didContractAdd
                };
                ethrDidResolver = (0, ethr_did_resolver_1.getResolver)(providerConfig);
                didResolver = new did_resolver_1.Resolver(ethrDidResolver);
                return [4 /*yield*/, (0, did_jwt_vc_1.verifyCredential)(vc, didResolver)
                    // return verifiedVc
                    // console.log(verifiedVc.payload.iss)
                    // console.log(issuerDid)
                ];
            case 1:
                verifiedVc = _a.sent();
                // return verifiedVc
                // console.log(verifiedVc.payload.iss)
                // console.log(issuerDid)
                if (verifiedVc.payload.iss !== issuerDid) {
                    console.log("False");
                    return [2 /*return*/, "False"];
                }
                else {
                    // console.log("eee")
                    console.log("True");
                    return [2 /*return*/, "True"];
                }
                return [3 /*break*/, 3];
            case 2:
                e_1 = _a.sent();
                return [2 /*return*/, e_1];
            case 3: return [2 /*return*/];
        }
    });
}); };
// console.log(didContractAdd)
// console.log(verifyVC("eyJhbGciOiJFUzI1NkstUiIsInR5cCI6IkpXVCJ9.eyJ2YyI6eyJAY29udGV4dCI6WyJodHRwczovL3d3dy53My5vcmcvMjAxOC9jcmVkZW50aWFscy92MSJdLCJ0eXBlIjpbIlZlcmlmaWFibGVDcmVkZW50aWFsIl0sImNyZWRlbnRpYWxTdWJqZWN0Ijp7InRyYWluZXIiOnsiZGlkIjoiZGlkOmV0aHI6Z2FuYWNoZToweDNFNWU5MTExQWU4ZUI3OEZlMUNDM2JiODkxNWQ1RDQ2MUYzRWY5QTkifX19LCJzdWIiOiJkaWQ6ZXRocjpnYW5hY2hlOmRpZDpldGhyOmdhbmFjaGU6MHgzRTVlOTExMUFlOGVCNzhGZTFDQzNiYjg5MTVkNUQ0NjFGM0VmOUE5IiwibmJmIjoxNTYyOTUwMjgyLCJpc3MiOiJkaWQ6ZXRocjpnYW5hY2hlOjB4MEQzOGU2NTNlQzI4YmRlYTVBMjI5NmZENTk0MGFhQjJEMEI4ODc1YyJ9.lVbydxlN6Mn0ngelfdsem8S96s6qmuceA-iVmv5UO1XQfPFZB2DtiEIm5iAzTYfcKx-AGAdv9jvqEfIlPeyU4gE","did:ethr:ganache:did:ethr:ganache:0x3E5e9111Ae8eB78Fe1CC3bb8915d5D461F3Ef9A9")
// )
module.exports = {
    createIssuerDID: createIssuerDID,
    createHolderDID: createHolderDID,
    issueVC: issueVC,
    verifyVC: verifyVC
};
