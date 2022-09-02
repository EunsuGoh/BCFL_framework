
export type CIJSON = {
    /**
     * Initialise Object's values (date and file)
     */
    init(): void
    /**
     * Extract schema in json
     * @param schema Schema you want extract in JSON object
     */
    extract(schema: string): CIJSON
    /**
     * Set data in an JSON value
     * @param key Key of JSON Object
     * @param value The value you want to set
     */
    set(key: string, value: any): CIJSON
    /**
    * Delete Key into JSON data
    * @param key Key you want to delete
    */
    remove(key: string): CIJSON
    /**
    * Push data in JSON Object
    * @param key Key of JSON Object
    * @param values The values you want to push
    */
    pushTo(key: string, ...values: any[]): CIJSON
    /**
    * Push data in JSON Object
    * @param key Key of JSON Object
    */
    popTo(key: string): CIJSON
    /**
     * Save final JSON in FIle
     * @param output Output file to save JSON Object
     */
    save(output: string): void
    /**
     * Get data JSON as Object
     */
    getData(): object
    /**
    * Get values of JSON
    */
    getValues(): Array<string> | Array<number> | Array<object>
    /**
    * Get keys of JSON
    */
    getKeys(): Array<string> | Array<number>
}


/**
 * Create json object
 * @param object Object to initialize json
 */
export function create(object: object): CIJSON
/**
 * Open a JSON file
 * @param input JSON File you want open
 */
export function open(input: string): CIJSON


